-- Lookup tables shared by the accuracy and performance databases.
-- One row per CI / nightly / weekly / acceptance / on-demand run; shards of a run share it.
CREATE TABLE IF NOT EXISTS run (
    id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL UNIQUE,
    type text NOT NULL CHECK (type IN ('ci', 'nightly', 'weekly', 'acceptance', 'on-demand')),
    started_at timestamptz NOT NULL DEFAULT now(),
    url text
);
CREATE INDEX IF NOT EXISTS run_type_started_idx ON run (type, started_at DESC);

-- NULLS NOT DISTINCT lets identical environments with unset optional fields reuse one row.
CREATE TABLE IF NOT EXISTS host (
    id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    hostname text NOT NULL,
    cpu text,
    gpu text,
    memory text NOT NULL,
    disk_root text NOT NULL,
    disk_home text,
    disk_opt text,
    disk_var text,
    os text NOT NULL,
    kernel text NOT NULL,
    compute_runtime text,
    igc text,
    l0 text,
    gcc text,
    UNIQUE NULLS NOT DISTINCT (hostname, cpu, gpu, memory, disk_root, disk_home, disk_opt, disk_var,
                               os, kernel, compute_runtime, igc, l0, gcc)
);
CREATE TABLE IF NOT EXISTS container (
    id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL,
    os text NOT NULL,
    compute_runtime text,
    igc text,
    l0 text,
    gcc text,
    UNIQUE NULLS NOT DISTINCT (name, os, compute_runtime, igc, l0, gcc)
);
CREATE TABLE IF NOT EXISTS torch (
    id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL,
    version text NOT NULL,
    commit text NOT NULL,
    repo text,
    torchvision text,
    torchaudio text,
    torchbench text,
    transformers text,
    timm text,
    pytest text,
    UNIQUE NULLS NOT DISTINCT (name, version, commit, repo, torchvision, torchaudio, torchbench,
                               transformers, timm, pytest)
);
CREATE TABLE IF NOT EXISTS python (
    id smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    version text NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS device (
    id smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS backend (
    id smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS suite (
    id smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS dtype (
    id smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS mode (
    id smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS model (
    id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL UNIQUE
);
-- Case outcome: "pass" or the failure reason (e.g. fail_accuracy, eager_fail_to_run).
CREATE TABLE IF NOT EXISTS result (
    id smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS issue (
    id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    repo text NOT NULL,
    number integer NOT NULL,
    title text,
    body text,
    created_by text,
    created_at timestamptz,
    UNIQUE (repo, number)
);
