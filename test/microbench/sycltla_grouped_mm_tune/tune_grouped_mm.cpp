/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// Standalone tuner for the SYCL-TLA grouped GEMM (no PyTorch). Runs the six
// grouped GEMMs of one DeepSeek-V3 16B MoE layer per EP rank, with routed row
// counts, against every compiled tile config and a list of persistent grid
// sizes, and reports the best config per op next to the production default.
// The kernels are the production ones (grouped_mm_kernel_impl.h), so a winning
// config can be moved into torch-xpu-ops as is. See README.md.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "tune_registry.h"

using bf16 = sycl::ext::oneapi::bfloat16;
using sycltla::grouped_gemm::Params;

namespace {

// ---------------------------------------------------------------- model ---
constexpr int kDim = 2048;        // DeepSeek-V3 16B hidden size
constexpr int kMoeHidden = 1408;  // expert FFN width (w13 is fused: 2 * 1408)
constexpr int kExperts = 64;
constexpr int kTopK = 6;

struct Opts {
  int ep = 8;
  int tokens = 8192;              // tokens per rank per microbatch (4 x 2048)
  std::vector<std::string> routings{"balanced", "uniform", "skewed", "hot", "sparse"};
  std::vector<std::string> ops{"w13_fwd", "w2_fwd", "w13_dX", "w2_dX", "w13_dW", "w2_dW"};
  std::vector<std::string> configs;  // empty = all compiled
  std::vector<int> wgs;              // empty = auto list from core count
  std::vector<int> sweep;            // rows per expert, equal across experts
  int iters = 10;
  int warmup = 2;
  bool check = true;
  std::string csv;
  std::string cfg_dir;               // where libtune_cfg_*.so live (default: next to the binary)
  bool list = false;
  unsigned seed = 0;
};

std::vector<const tune::Candidate*> load_candidates(const std::string& dir) {
  std::vector<std::string> files;
  if (DIR* d = opendir(dir.c_str())) {
    while (dirent* e = readdir(d)) {
      std::string n = e->d_name;
      if (n.rfind("libtune_cfg_", 0) == 0 && n.size() > 3 && n.substr(n.size() - 3) == ".so")
        files.push_back(dir + "/" + n);
    }
    closedir(d);
  }
  std::sort(files.begin(), files.end());
  std::vector<const tune::Candidate*> out;
  for (auto& f : files) {
    void* h = dlopen(f.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "skip %s: %s\n", f.c_str(), dlerror()); continue; }
    auto fn = reinterpret_cast<tune::CandidateFn>(dlsym(h, tune::kCandidateSymbol));
    if (!fn) { std::fprintf(stderr, "skip %s: no %s\n", f.c_str(), tune::kCandidateSymbol); continue; }
    out.push_back(fn());
  }
  // production configs first, then the rest by name
  std::stable_sort(out.begin(), out.end(), [](auto* a, auto* b) {
    return (a->name[0] == 'p') > (b->name[0] == 'p');
  });
  return out;
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  size_t start = 0;
  while (start <= s.size()) {
    size_t end = s.find(',', start);
    if (end == std::string::npos) end = s.size();
    if (end > start) out.push_back(s.substr(start, end - start));
    start = end + 1;
  }
  return out;
}

void usage() {
  std::printf(
      "tune_grouped_mm [options]\n"
      "  --ep N             expert-parallel degree (default 8 -> 8 local experts)\n"
      "  --tokens N         tokens per rank per microbatch (default 8192)\n"
      "  --routings a,b     balanced,uniform,skewed,hot,sparse (default all; '' none)\n"
      "  --sweep r1,r2      also run equal rows/expert cases (e.g. 64,256,1024,6144)\n"
      "  --ops a,b          w13_fwd,w2_fwd,w13_dX,w2_dX,w13_dW,w2_dW (default all)\n"
      "  --configs a,b      subset of compiled configs (default all; --list to see)\n"
      "  --wgs a,b          persistent work-group counts (default: cores x {1,2,3,4})\n"
      "  --iters N --warmup N --seed N\n"
      "  --no-check         skip the CPU reference check\n"
      "  --csv FILE         write every measurement\n"
      "  --cfg-dir DIR      directory with libtune_cfg_*.so (default: binary's dir)\n"
      "  --list             list compiled configs and exit\n");
}

Opts parse(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { usage(); std::exit(2); }
      return argv[++i];
    };
    if (a == "--ep") o.ep = std::stoi(next());
    else if (a == "--tokens") o.tokens = std::stoi(next());
    else if (a == "--routings") o.routings = split(next());
    else if (a == "--ops") o.ops = split(next());
    else if (a == "--configs") o.configs = split(next());
    else if (a == "--wgs") { o.wgs.clear(); for (auto& s : split(next())) o.wgs.push_back(std::stoi(s)); }
    else if (a == "--sweep") { o.sweep.clear(); for (auto& s : split(next())) o.sweep.push_back(std::stoi(s)); }
    else if (a == "--iters") o.iters = std::stoi(next());
    else if (a == "--warmup") o.warmup = std::stoi(next());
    else if (a == "--seed") o.seed = unsigned(std::stoul(next()));
    else if (a == "--no-check") o.check = false;
    else if (a == "--csv") o.csv = next();
    else if (a == "--cfg-dir") o.cfg_dir = next();
    else if (a == "--list") o.list = true;
    else { usage(); std::exit(a == "-h" || a == "--help" ? 0 : 2); }
  }
  return o;
}

// -------------------------------------------------------------- routing ---
// Per-local-expert row counts for EP rank 0, following torchtitan's top-k
// router: every one of the EP * T global tokens picks kTopK distinct experts.
std::vector<int> topk_counts(int g_tokens, int e_local, std::mt19937& rng,
                             const std::vector<float>& bias, int force_expert) {
  std::uniform_real_distribution<float> u(0.f, 1.f);
  std::vector<int> counts(kExperts, 0);
  std::vector<std::pair<float, int>> s(kExperts);
  for (int t = 0; t < g_tokens; ++t) {
    for (int e = 0; e < kExperts; ++e) {
      float g = -std::log(-std::log(std::max(u(rng), 1e-12f)));  // Gumbel noise
      s[e] = {bias.empty() ? u(rng) : bias[e] + g, e};
    }
    if (force_expert >= 0) s[force_expert].first = 1e9f;
    std::partial_sort(s.begin(), s.begin() + kTopK, s.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });
    for (int k = 0; k < kTopK; ++k) counts[s[k].second]++;
  }
  return std::vector<int>(counts.begin(), counts.begin() + e_local);
}

std::vector<int> route(const std::string& kind, int g_tokens, int e_local, std::mt19937& rng) {
  if (kind == "balanced")
    return std::vector<int>(e_local, g_tokens * kTopK / kExperts);
  if (kind == "uniform") return topk_counts(g_tokens, e_local, rng, {}, -1);
  if (kind == "skewed") {  // Zipf-like popularity in random order
    std::vector<int> rank(kExperts);
    std::iota(rank.begin(), rank.end(), 1);
    std::shuffle(rank.begin(), rank.end(), rng);
    std::vector<float> bias(kExperts);
    for (int e = 0; e < kExperts; ++e) bias[e] = -1.2f * std::log(float(rank[e]));
    return topk_counts(g_tokens, e_local, rng, bias, -1);
  }
  if (kind == "hot") return topk_counts(g_tokens, e_local, rng, {}, 0);
  if (kind == "sparse") {
    auto c = topk_counts(g_tokens, e_local, rng, {}, -1);
    std::vector<int> idx(e_local);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    for (int i = 0; i < e_local / 2; ++i) c[idx[i]] = 0;  // half the experts idle
    return c;
  }
  std::fprintf(stderr, "unknown routing %s\n", kind.c_str());
  std::exit(2);
}

// ------------------------------------------------------------------ ops ---
struct OpDef {
  std::string name;
  bool k_grouped;
  char layout_b;  // M-grouped only
  int a_cols;     // M-grouped: K. K-grouped: the fixed output M (width of the activation)
  int n;          // N
  int a_buf, b_buf;  // which host/device buffer (0 x, 1 h, 2 dg, 3 dy, 4 w13, 5 w2)
};

const std::vector<OpDef>& op_defs() {
  static const std::vector<OpDef> d = {
      {"w13_fwd", false, 'R', kDim, 2 * kMoeHidden, 0, 4},
      {"w2_fwd", false, 'R', kMoeHidden, kDim, 1, 5},
      {"w13_dX", false, 'C', 2 * kMoeHidden, kDim, 2, 4},
      {"w2_dX", false, 'C', kDim, kMoeHidden, 3, 5},
      {"w13_dW", true, 'C', 2 * kMoeHidden, kDim, 2, 0},
      {"w2_dW", true, 'C', kDim, kMoeHidden, 3, 1},
  };
  return d;
}

int buf_cols(int b) {
  switch (b) {
    case 0: return kDim;
    case 1: return kMoeHidden;
    case 2: return 2 * kMoeHidden;
    case 3: return kDim;
    default: return 0;
  }
}

// ------------------------------------------------------------- buffers ---
struct Buffers {
  sycl::queue& q;
  int m_max, e_local;
  std::vector<std::vector<bf16>> host;  // 0 x, 1 h, 2 dg, 3 dy, 4 w13, 5 w2
  std::vector<bf16*> dev;
  bf16* out = nullptr;
  size_t out_elems = 0;
  int32_t* offs = nullptr;

  Buffers(sycl::queue& q_, int m_max_, int e_local_, unsigned seed)
      : q(q_), m_max(m_max_), e_local(e_local_), host(6), dev(6, nullptr) {
    const size_t sizes[6] = {
        size_t(m_max) * kDim, size_t(m_max) * kMoeHidden, size_t(m_max) * 2 * kMoeHidden,
        size_t(m_max) * kDim, size_t(e_local) * 2 * kMoeHidden * kDim,
        size_t(e_local) * kDim * kMoeHidden};
    uint32_t x = 0x9e3779b9u ^ seed;
    for (int b = 0; b < 6; ++b) {
      host[b].resize(sizes[b]);
      for (auto& v : host[b]) {  // xorshift: fast and reproducible
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        v = bf16((float(x >> 8) * (1.0f / 16777216.0f) - 0.5f) * 1.0f);
      }
      dev[b] = sycl::malloc_device<bf16>(sizes[b] + 64, q);
      q.memcpy(dev[b], host[b].data(), sizes[b] * sizeof(bf16));
    }
    out_elems = std::max(size_t(m_max) * 2 * kMoeHidden,
                         size_t(e_local) * 2 * kMoeHidden * kDim);
    out = sycl::malloc_device<bf16>(out_elems + 64, q);
    offs = sycl::malloc_device<int32_t>(e_local, q);
    q.wait();
  }
  ~Buffers() {
    for (auto* p : dev) sycl::free(p, q);
    sycl::free(out, q);
    sycl::free(offs, q);
  }
};

Params make_params(const OpDef& op, const Buffers& b, int total_rows) {
  Params p;
  p.a = b.dev[op.a_buf];
  p.b = b.dev[op.b_buf];
  p.d = b.out;
  p.offs = b.offs;
  p.num_groups = b.e_local;
  p.N = op.n;
  if (!op.k_grouped) {
    p.M = total_rows;
    p.K = op.a_cols;
    p.lda = op.a_cols;
    p.ldd = op.n;
    // weights are stored [E, O, I]; 'R' reads (n, k) at n * I + k, 'C' at n + k * O
    p.ldb = (op.layout_b == 'R') ? op.a_cols : op.n;
    p.group_stride_b = int64_t(op.a_cols) * op.n;
  } else {
    p.M = op.a_cols;
    p.K = total_rows;
    p.lda = op.a_cols;           // activation [R, M] row-major viewed as (M, R) col-major
    p.ldb = buf_cols(op.b_buf);  // [R, N] row-major
    p.ldd = op.n;
    p.group_stride_d = int64_t(op.a_cols) * op.n;
  }
  return p;
}

void launch(const tune::Candidate& c, const OpDef& op, sycl::queue& q, const Params& p, int wgs) {
  if (op.k_grouped) c.k_grouped(q, p, wgs);
  else if (op.layout_b == 'R') c.m_rowB(q, p, wgs);
  else c.m_colB(q, p, wgs);
}

// CPU reference on a few sampled output rows.
struct Sample { int g, row; };

std::vector<float> reference_row(const OpDef& op, const Buffers& b, const std::vector<int>& starts,
                                 const std::vector<int>& counts, const Sample& s) {
  std::vector<float> r(op.n, 0.f);
  const auto& A = b.host[op.a_buf];
  const auto& B = b.host[op.b_buf];
  if (!op.k_grouped) {
    const int K = op.a_cols;
    const bf16* a = &A[size_t(s.row) * K];
    const bf16* w = &B[size_t(s.g) * K * op.n];
    for (int n = 0; n < op.n; ++n) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) {
        float bv = (op.layout_b == 'R') ? float(w[size_t(n) * K + k]) : float(w[size_t(k) * op.n + n]);
        acc += float(a[k]) * bv;
      }
      r[n] = acc;
    }
  } else {
    const int Mcols = op.a_cols, bcols = buf_cols(op.b_buf);
    for (int t = starts[s.g]; t < starts[s.g] + counts[s.g]; ++t) {
      const float av = float(A[size_t(t) * Mcols + s.row]);
      const bf16* brow = &B[size_t(t) * bcols];
      for (int n = 0; n < op.n; ++n) r[n] += av * float(brow[n]);
    }
  }
  return r;
}

bool check_rows(const OpDef& op, Buffers& b, const std::vector<int>& starts,
                const std::vector<int>& counts, const std::vector<Sample>& samples,
                const std::vector<std::vector<float>>& refs, double& max_err) {
  std::vector<bf16> row(op.n);
  bool ok = true;
  max_err = 0;
  for (size_t i = 0; i < samples.size(); ++i) {
    const auto& s = samples[i];
    size_t off = op.k_grouped ? (size_t(s.g) * op.a_cols + s.row) * op.n : size_t(s.row) * op.n;
    b.q.memcpy(row.data(), b.out + off, op.n * sizeof(bf16)).wait();
    for (int n = 0; n < op.n; ++n) {
      double err = std::fabs(double(float(row[n])) - double(refs[i][n]));
      double tol = 0.02 * std::fabs(double(refs[i][n])) + 0.1;
      if (!(err <= tol)) ok = false;
      max_err = std::max(max_err, err);
    }
  }
  return ok;
}

double flops_of(const OpDef& op, int total_rows) {
  return 2.0 * double(total_rows) * op.a_cols * op.n;
}

} // namespace

int main(int argc, char** argv) {
  Opts o = parse(argc, argv);
  if (o.cfg_dir.empty()) {
    std::string self = argv[0];
    auto slash = self.rfind('/');
    o.cfg_dir = slash == std::string::npos ? "." : self.substr(0, slash);
  }
  auto all = load_candidates(o.cfg_dir);
  if (all.empty()) { std::fprintf(stderr, "no libtune_cfg_*.so in %s\n", o.cfg_dir.c_str()); return 1; }
  if (o.list) {
    for (auto* c : all) std::printf("%-16s %-28s %d threads\n", c->name.c_str(), c->desc.c_str(), c->threads);
    return 0;
  }

  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
  auto dev = q.get_device();
  namespace gg = sycltla::grouped_gemm;
  const int cores = gg::xe_core_count(dev);
  if (o.wgs.empty()) o.wgs = {cores, 2 * cores, 3 * cores, 4 * cores};

  std::vector<const tune::Candidate*> cands;
  for (auto* c : all)
    if (o.configs.empty() || std::find(o.configs.begin(), o.configs.end(), c->name) != o.configs.end())
      cands.push_back(c);

  const int e_local = kExperts / o.ep;
  const int g_tokens = o.ep * o.tokens;
  std::mt19937 rng(o.seed);

  // Build the cases.
  struct Case { std::string name; std::vector<int> counts; };
  std::vector<Case> cases;
  for (auto& r : o.routings) cases.push_back({r, route(r, g_tokens, e_local, rng)});
  for (int m : o.sweep) cases.push_back({"rows" + std::to_string(m), std::vector<int>(e_local, m)});
  int m_max = 1;
  for (auto& c : cases) m_max = std::max(m_max, std::accumulate(c.counts.begin(), c.counts.end(), 0));

  std::printf("device: %s   %d Xe cores   production default: choose_tile() per op\n",
              dev.get_info<sycl::info::device::name>().c_str(), cores);
  std::printf("DSv3 16B: EP=%d, %d local experts, %d tokens/rank, %d global tokens, top-%d\n",
              o.ep, e_local, o.tokens, g_tokens, kTopK);
  std::printf("configs:");
  for (auto* c : cands) std::printf(" %s(%s)", c->name.c_str(), c->desc.c_str());
  std::printf("\nwork-groups:");
  for (int w : o.wgs) std::printf(" %d", w);
  std::printf("\nallocating for %d rows...\n", m_max);
  std::fflush(stdout);
  Buffers b(q, m_max, e_local, o.seed);

  FILE* csv = o.csv.empty() ? nullptr : std::fopen(o.csv.c_str(), "w");
  if (csv) std::fprintf(csv, "case,op,config,desc,wgs,total_rows,rows_min,rows_max,ms,tflops,ok\n");

  std::map<std::string, const OpDef*> ops;
  for (auto& d : op_defs()) ops[d.name] = &d;

  for (auto& cs : cases) {
    const int total = std::accumulate(cs.counts.begin(), cs.counts.end(), 0);
    if (total == 0) continue;
    std::vector<int32_t> offs(e_local);
    std::vector<int> starts(e_local, 0);
    int acc = 0;
    for (int e = 0; e < e_local; ++e) { starts[e] = acc; acc += cs.counts[e]; offs[e] = acc; }
    q.memcpy(b.offs, offs.data(), e_local * sizeof(int32_t)).wait();
    const int cmin = *std::min_element(cs.counts.begin(), cs.counts.end());
    const int cmax = *std::max_element(cs.counts.begin(), cs.counts.end());
    std::printf("\n== case %s: rows/expert %d..%d, total %d\n", cs.name.c_str(), cmin, cmax, total);
    std::printf("  %-8s %-16s %5s %9s %8s %6s   %s\n", "op", "best config", "wgs", "best ms", "TFLOPS", "ok",
                "vs production default");

    for (auto& on : o.ops) {
      auto it = ops.find(on);
      if (it == ops.end()) { std::fprintf(stderr, "unknown op %s\n", on.c_str()); return 2; }
      const OpDef& op = *it->second;
      Params p = make_params(op, b, total);
      // production choice for this call: choose_tile() and its default grid
      const int ptile = gg::choose_tile(dev, p);
      const std::string prod = std::string("p") + gg::tile_name(ptile);
      const int prod_wgs = gg::default_num_workgroups(dev, ptile);

      // reference rows
      std::vector<Sample> samples;
      std::vector<std::vector<float>> refs;
      if (o.check) {
        std::mt19937 srng(123);
        for (int i = 0; i < 6; ++i) {
          int g;
          do { g = int(srng() % e_local); } while (cs.counts[g] == 0);
          int row = op.k_grouped ? int(srng() % op.a_cols) : starts[g] + int(srng() % cs.counts[g]);
          samples.push_back({g, row});
          refs.push_back(reference_row(op, b, starts, cs.counts, samples.back()));
        }
      }

      double best_ms = 1e30, prod_ms = -1;
      std::string best_cfg;
      int best_w = 0;
      bool all_ok = true;
      for (auto* c : cands) {
        for (int w : o.wgs) {
          for (int i = 0; i < o.warmup; ++i) launch(*c, op, q, p, w);
          q.wait();
          auto t0 = std::chrono::steady_clock::now();
          for (int i = 0; i < o.iters; ++i) launch(*c, op, q, p, w);
          q.wait();
          double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / o.iters;
          bool ok = true;
          double err = 0;
          if (o.check) ok = check_rows(op, b, starts, cs.counts, samples, refs, err);
          all_ok &= ok;
          double tf = flops_of(op, total) / (ms * 1e-3) / 1e12;
          if (csv)
            std::fprintf(csv, "%s,%s,%s,%s,%d,%d,%d,%d,%.4f,%.2f,%d\n", cs.name.c_str(), op.name.c_str(),
                         c->name.c_str(), c->desc.c_str(), w, total, cmin, cmax, ms, tf, int(ok));
          if (ok && ms < best_ms) { best_ms = ms; best_cfg = c->name; best_w = w; }
          if (c->name == prod && w == prod_wgs) prod_ms = ms;
          if (!ok) std::printf("     MISMATCH %s wgs=%d max_err=%.3g\n", c->name.c_str(), w, err);
        }
      }
      double best_tf = flops_of(op, total) / (best_ms * 1e-3) / 1e12;
      char vs[96] = "production config not in run";
      if (prod_ms > 0)
        std::snprintf(vs, sizeof(vs), "%s/%d: %.3f ms (%.1f TFLOPS), best is %.1f%% faster", prod.c_str(), prod_wgs,
                      prod_ms, flops_of(op, total) / (prod_ms * 1e-3) / 1e12, 100.0 * (prod_ms / best_ms - 1.0));
      std::printf("  %-8s %-16s %5d %9.3f %8.1f %6s   %s\n", op.name.c_str(), best_cfg.c_str(), best_w, best_ms,
                  best_tf, all_ok ? "yes" : "NO", vs);
      std::fflush(stdout);
    }
  }
  if (csv) { std::fclose(csv); std::printf("\nwrote %s\n", o.csv.c_str()); }
  return 0;
}
