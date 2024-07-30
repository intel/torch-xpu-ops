#pragma once

#include <ATen/Context.h>

namespace at::xpu {

#pragma once

// define null_type
struct null_type {};

// null_type comparisons
inline bool operator==(const null_type&, const null_type&) {
  return true;
}

inline bool operator>=(const null_type&, const null_type&) {
  return true;
}

inline bool operator<=(const null_type&, const null_type&) {
  return true;
}

inline bool operator!=(const null_type&, const null_type&) {
  return false;
}

inline bool operator<(const null_type&, const null_type&) {
  return false;
}

inline bool operator>(const null_type&, const null_type&) {
  return false;
}

// forward declaration for tuple
template <
    class T0 = null_type,
    class T1 = null_type,
    class T2 = null_type,
    class T3 = null_type,
    class T4 = null_type,
    class T5 = null_type,
    class T6 = null_type,
    class T7 = null_type,
    class T8 = null_type,
    class T9 = null_type>
class tuple;

template <typename T1, typename T2>
inline void swap(T1& a, T2& b) {
  T1 temp = a;
  a = b;
  b = temp;
}

template <typename T1, typename T2>
struct pair {
  typedef T1 first_type;
  typedef T2 second_type;

  first_type first;
  second_type second;

  // default constructor
  pair(void) : first(), second() {}

  inline pair(const T1& x, const T2& y) : first(x), second(y) {}

  template <typename U1, typename U2>
  inline pair(const pair<U1, U2>& p) : first(p.first), second(p.second) {}

  template <typename U1, typename U2>
  pair(const std::pair<U1, U2>& p) : first(p.first), second(p.second) {}

  inline void swap(pair& p) {
    swap(first, p.first);
    swap(second, p, second);
  }
};

template <typename T1, typename T2>
bool operator==(const pair<T1, T2>& x, const pair<T1, T2>& y) {
  return x.first == y.first && x.second == y.second;
}

template <typename T1, typename T2>
inline bool operator<(const pair<T1, T2>& x, const pair<T1, T2>& y) {
  return x.first < y.first || (!(y.first < x.first) && x.second < y.second);
}

template <typename T1, typename T2>
inline bool operator!=(const pair<T1, T2>& x, const pair<T1, T2>& y) {
  return !(x == y);
}

template <typename T1, typename T2>
inline bool operator>(const pair<T1, T2>& x, const pair<T1, T2>& y) {
  return y < x;
}

template <typename T1, typename T2>
bool operator<=(const pair<T1, T2>& x, const pair<T1, T2>& y) {
  return !(y < x);
}

template <typename T1, typename T2>
bool operator>=(const pair<T1, T2>& x, const pair<T1, T2>& y) {
  return !(x < y);
}

template <typename T1, typename T2>
inline void swap(pair<T1, T2>& x, pair<T1, T2>& y) {
  return x.swap(y);
}

template <typename T1, typename T2>
inline pair<T1, T2> make_pair(T1 x, T2 y) {
  return pair<T1, T2>(x, y);
}

template <size_t N, class T>
struct tuple_element;

template <size_t N, class T>
struct tuple_element_impl {
 private:
  typedef typename T::tail_type Next;

 public:
  typedef typename tuple_element_impl<N - 1, Next>::type type;
};

template <class T>
struct tuple_element_impl<0, T> {
  typedef typename T::head_type type;
};

template <size_t N, class T>
struct tuple_element<N, T const> {
  using type =
      typename std::add_const<typename tuple_element<N, T>::type>::type;
};

template <size_t N, class T>
struct tuple_element<N, T volatile> {
  using type =
      typename std::add_volatile<typename tuple_element<N, T>::type>::type;
};

template <size_t N, class T>
struct tuple_element<N, T const volatile> {
  using type = typename std::add_cv<typename tuple_element<N, T>::type>::type;
};

template <size_t N, class T>
struct tuple_element {
  using type = typename tuple_element_impl<N, T>::type;
};

template <typename T1, typename T2>
struct tuple_element<0, pair<T1, T2>> {
  typedef T1 type;
};

template <typename T1, typename T2>
struct tuple_element<1, pair<T1, T2>> {
  typedef T2 type;
};

// forward declaration of tuple_size
template <class T>
struct tuple_size;

template <class T>
struct tuple_size<T const> : public tuple_size<T> {};

template <class T>
struct tuple_size<T volatile> : public tuple_size<T> {};

template <class T>
struct tuple_size<T const volatile> : public tuple_size<T> {};

template <class T>
struct tuple_size {
  static const int value = 1 + tuple_size<typename T::tail_type>::value;
};

// specializations for tuple_size
template <>
struct tuple_size<tuple<>> {
  static const int value = 0;
};

template <>
struct tuple_size<null_type> {
  static const int value = 0;
};

template <typename T1, typename T2>
struct tuple_size<pair<T1, T2>> {
  static const unsigned int value = 2;
};

namespace detail {

template <int N, typename Pair>
struct pair_get {};

template <typename Pair>
struct pair_get<0, Pair> {
  inline const typename tuple_element<0, Pair>::type& operator()(
      const Pair& p) const {
    return p.first;
  }

  inline typename tuple_element<0, Pair>::type& operator()(Pair& p) const {
    return p.first;
  }
};

template <typename Pair>
struct pair_get<1, Pair> {
  inline const typename tuple_element<1, Pair>::type& operator()(
      const Pair& p) const {
    return p.second;
  }

  inline typename tuple_element<1, Pair>::type& operator()(Pair& p) const {
    return p.second;
  }
};

} // namespace detail

template <unsigned int N, typename T1, typename T2>
inline typename tuple_element<N, pair<T1, T2>>::type& get(pair<T1, T2>& p) {
  return detail::pair_get<N, pair<T1, T2>>()(p);
}

template <unsigned int N, typename T1, typename T2>
inline const typename tuple_element<N, pair<T1, T2>>::type& get(
    const pair<T1, T2>& p) {
  return detail::pair_get<N, pair<T1, T2>>()(p);
}

} // namespace at::xpu
