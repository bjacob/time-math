/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// How many significant digits we really want to print for all those floats
const int lowprecision = 3;

// Minimum time in seconds that our payload needs to run to give a useful timing
const double minimum_measurement_time = 0.1;

// How many timings we run before we pick the best one
const int measurement_repetitions = 4;

// Threshold to report a worst-to-best timing ratio as something to worry about.
// Our ambient noise level is around 1.05 with current values of
// minimum_measurement_time and of measurement_repetitions.
const double worst_to_best_ratio_alert_threshold = 1.3;

#include <cstdint>
#include <iostream>
#include <ctime>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <cassert>

#ifdef X86_ENABLE_DAZ
#include <xmmintrin.h>
#define _MM_DENORMALS_ZERO_MASK   0x0040
#define _MM_DENORMALS_ZERO_ON     0x0040
#define _MM_SET_DENORMALS_ZERO_MODE(mode) \
  (_mm_setcsr((_mm_getcsr() & ~_MM_DENORMALS_ZERO_MASK) | (mode)))
#define _MM_GET_DENORMALS_ZERO_MODE() \
  (_mm_getcsr() & _MM_DENORMALS_ZERO_MASK)
#endif

#ifdef X86_ENABLE_FTZ
#include <xmmintrin.h>
#endif

// some versions of GCC don't properly optimize code in the main() function, so it is
// good practice to prevent benchmark code from being inlined into main().
#if (defined __GNUC__)
#define DONT_INLINE __attribute__((noinline))
#elif (defined _MSC_VER)
#define DONT_INLINE __declspec(noinline)
#else
#define DONT_INLINE
#endif

using namespace std;

// we record alerts to print an alerts summary at the end.
vector<string> alerts;

struct lowprecisionstringstream : public stringstream {
  lowprecisionstringstream() {
    precision(lowprecision);
  }
};

double time() {
  timespec t;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
  return double(t.tv_sec) + 1e-9 * double(t.tv_nsec);
}

template<typename scalar>
struct scalar_type_info
{};

template<>
struct scalar_type_info<float>
{
  static string name() { return "float"; }
};

template<>
struct scalar_type_info<double>
{
  static string name() { return "double"; }
};

template<>
struct scalar_type_info<int8_t>
{
  static string name() { return "int8"; }
};

template<>
struct scalar_type_info<int16_t>
{
  static string name() { return "int16"; }
};

template<>
struct scalar_type_info<int32_t>
{
  static string name() { return "int32"; }
};

template<>
struct scalar_type_info<int64_t>
{
  static string name() { return "int64"; }
};

template<typename scalar>
scalar as_number(scalar x)
{
  return x;
}

int as_number(int8_t x)
{
  return x;
}

template<typename from_scalar, typename to_scalar>
struct conversion
{
  volatile from_scalar x;
  mutable volatile to_scalar result;

  conversion(from_scalar _x = 0)
    : x(_x)
    , result(0)
  {}

  void run() const {
    result = static_cast<to_scalar>(x);
  }

  static string generic_name() {
    return string("conversion from ") + scalar_type_info<from_scalar>::name()
           + " to " + scalar_type_info<to_scalar>::name();
  }

  string name() const {
    lowprecisionstringstream result_stream;
    result_stream << generic_name() << " of " << as_number(x);
    return result_stream.str();
  }
};

template<typename scalar>
struct exponential
{
  volatile scalar x;
  mutable volatile scalar result;

  exponential(scalar _x = 0)
    : x(_x)
    , result(0)
  {}

  void run() const {
    result = std::exp(x);
  }

  static string generic_name() {
    return string("std::exp<") + scalar_type_info<scalar>::name() + ">";
  }

  string name() const {
    lowprecisionstringstream result_stream;
    result_stream << generic_name() << "(" << as_number(x) << ")";
    return result_stream.str();
  }
};

template<typename scalar>
struct cosine
{
  volatile scalar x;
  mutable volatile scalar result;

  cosine(scalar _x = 0)
    : x(_x)
    , result(0)
  {}

  void run() const {
    result = std::cos(x);
  }

  static string generic_name() {
    return string("std::cos<") + scalar_type_info<scalar>::name() + ">";
  }

  string name() const {
    lowprecisionstringstream result_stream;
    result_stream << generic_name() << "(" << as_number(x) << ")";
    return result_stream.str();
  }
};

template<typename scalar>
struct logarithm
{
  volatile scalar x;
  mutable volatile scalar result;

  logarithm(scalar _x = 0)
    : x(_x)
    , result(0)
  {}

  void run() const {
    result = std::log(x);
  }

  static string generic_name() {
    return string("std::log<") + scalar_type_info<scalar>::name() + ">";
  }

  string name() const {
    lowprecisionstringstream result_stream;
    result_stream << generic_name() << "(" << as_number(x) << ")";
    return result_stream.str();
  }
};

template<typename scalar>
struct arctangent
{
  volatile scalar x;
  mutable volatile scalar result;

  arctangent(scalar _x = 0)
    : x(_x)
    , result(0)
  {}

  void run() const {
    result = std::atan(x);
  }

  static string generic_name() {
    return string("std::atan<") + scalar_type_info<scalar>::name() + ">";
  }

  string name() const {
    lowprecisionstringstream result_stream;
    result_stream << generic_name() << "(" << as_number(x) << ")";
    return result_stream.str();
  }
};

template<typename scalar>
struct addition
{
  volatile scalar x, y;
  mutable volatile scalar result;

  addition(scalar _x = 0, scalar _y = 0)
    : x(_x)
    , y(_y)
    , result(0)
  {}

  void run() const {
    result = x + y;
  }

  static string generic_name() {
    return scalar_type_info<scalar>::name() + " addition";
  }

  string name() const {
    lowprecisionstringstream result_stream;
    result_stream << generic_name() << " of " << as_number(x) << " and " << as_number(y);
    return result_stream.str();
  }
};

template<typename scalar>
struct subtraction
{
  volatile scalar x, y;
  mutable volatile scalar result;

  subtraction(scalar _x = 0, scalar _y = 0)
    : x(_x)
    , y(_y)
    , result(0)
  {}

  void run() const {
    result = x - y;
  }

  static string generic_name() {
    return scalar_type_info<scalar>::name() + " subtraction";
  }

  string name() const {
    lowprecisionstringstream result_stream;
    result_stream << generic_name() << " of " << as_number(y) << " from " << as_number(x);
    return result_stream.str();
  }
};

template<typename scalar>
struct multiplication
{
  volatile scalar x, y;
  mutable volatile scalar result;

  multiplication(scalar _x = 0, scalar _y = 0)
    : x(_x)
    , y(_y)
    , result(0)
  {}

  void run() const {
    result = x * y;
  }

  static string generic_name() {
    return scalar_type_info<scalar>::name() + " multiplication";
  }

  string name() const {
    lowprecisionstringstream result_stream;
    result_stream << generic_name() << " of " << as_number(x) << " by " << as_number(y);
    return result_stream.str();
  }
};

template<typename scalar>
struct division
{
  volatile scalar x, y;
  mutable volatile scalar result;

  division(scalar _x = 0, scalar _y = 0)
    : x(_x)
    , y(_y)
    , result(0)
  {
    // avoid integer division by zero
    if (numeric_limits<scalar>::is_integer) {
      // apparently --- wtf? --- we also need to avoid -1, as with int32_t,
      // the integer overflow in int_min / -1 gives me a SIGFPE!
      if (y == 0 || y == -1) {
        y = 1;
      }
    }
  }

  void run() const {
    result = x / y;
  }

  static string generic_name() {
    return scalar_type_info<scalar>::name() + " division";
  }

  string name() const {
    lowprecisionstringstream result_stream;
    result_stream << generic_name() << " of " << as_number(x) << " by " << as_number(y);
    return result_stream.str();
  }
};

template<typename functor>
DONT_INLINE
double one_timing(const functor& f)
{
  uint64_t total_runs = 0;
  double time1 = time();
  double loop_start_time = time1;
  uint64_t unrolled_loop_iters = 1000; // start with a small value, we will probably increase it below
  while (loop_start_time < time1 + minimum_measurement_time) {
    const size_t loop_unroll = 4;
    for (uint64_t i = 0; i < unrolled_loop_iters; i++) {
      for (size_t j = 0; j < loop_unroll; j++) {
        f.run();
      }
    }
    total_runs += loop_unroll * unrolled_loop_iters;
    double loop_end_time = time();

    // check if we can multiply unrolled_loop_iters by 10 without it taking too much time.
    const double target_loop_time = minimum_measurement_time / 10;
    if (loop_end_time - loop_start_time < target_loop_time) {
      unrolled_loop_iters *= 10;
    }

    loop_start_time = loop_end_time;
  }
  double time_per_run = (loop_start_time - time1) / total_runs;
  return time_per_run;
}

template<typename functor>
DONT_INLINE
double best_timing(const functor& f)
{
  double best = 0;
  for (size_t i = 0; i < measurement_repetitions; i++) {
    double t = one_timing(f);
    if (i == 0 || t < best) {
      best = t;
    }
  }
  return best;
}

template<typename functor>
void report_functor_summary(double best_time, double worst_time,
                            const functor& best_functor,
                            const functor& worst_functor)
{
  cout << "  best time: " << best_time << " s for " << best_functor.name() << endl;
  cout << "  worst time: " << worst_time << " s for " << worst_functor.name() << endl;
  double ratio = worst_time / best_time;
  cout << "  worst/best ratio: " << ratio << endl;
  if (ratio > worst_to_best_ratio_alert_threshold) {
    lowprecisionstringstream alert;
    size_t alert_number = alerts.size() + 1;
    alert << "alert #" << alert_number
          << ": high worst/best ratio of " << ratio
          << " for " << functor::generic_name();
    alerts.push_back(alert.str());
    cout << "  " << alerts.back() << endl;
  }
}

template<typename functor, typename scalar>
void study_binary_functor(const vector<scalar>& scalars)
{
  cout << functor::generic_name() << endl;

  double best_time;
  double worst_time;
  functor best_functor;
  functor worst_functor;
  bool first_time = true;

  cout << "  timings:" << endl;

  for (typename vector<scalar>::const_iterator itx = scalars.begin();
       itx != scalars.end();
       ++itx)
  {
    for (typename vector<scalar>::const_iterator ity = scalars.begin();
         ity != scalars.end();
         ++ity)
    {
      functor f(*itx, *ity);
      double time = best_timing(f);
      cout << "    " << f.name() << " : " << time << " s" << endl;
      if (first_time || time < best_time) {
        best_time = time;
        best_functor = f;
      }
      if (first_time || time > worst_time) {
        worst_time = time;
        worst_functor = f;
      }
      first_time = false;
    }
  }

  assert(!first_time);
  report_functor_summary(best_time, worst_time, best_functor, worst_functor);
  cout << endl;
}

template<typename functor, typename scalar>
void study_unary_functor(const vector<scalar>& scalars)
{
  cout << functor::generic_name() << endl;

  double best_time;
  double worst_time;
  functor best_functor;
  functor worst_functor;
  bool first_time = true;

  cout << "  timings:" << endl;

  for (typename vector<scalar>::const_iterator itx = scalars.begin();
       itx != scalars.end();
       ++itx)
  {
    functor f(*itx);
    double time = best_timing(f);
    cout << "    " << f.name() << " : " << time << " s" << endl;
    if (first_time || time < best_time) {
      best_time = time;
      best_functor = f;
    }
    if (first_time || time > worst_time) {
      worst_time = time;
      worst_functor = f;
    }
    first_time = false;
  }

  assert(!first_time);
  report_functor_summary(best_time, worst_time, best_functor, worst_functor);
  cout << endl;
}

template<typename scalar>
struct other_floating_point_type
{};

template<>
struct other_floating_point_type<float>
{
  typedef double type;
};

template<>
struct other_floating_point_type<double>
{
  typedef float type;
};

template<typename scalar, bool is_integer = numeric_limits<scalar>::is_integer>
struct study_scalar_type_impl
{
};

template<typename scalar>
struct study_scalar_type_impl<scalar, true>
{
  static void run(const vector<scalar>& vals)
  {
    study_unary_functor<conversion<scalar, float>>(vals);
    study_unary_functor<conversion<scalar, double>>(vals);

    study_binary_functor<addition<scalar>>(vals);
    study_binary_functor<subtraction<scalar>>(vals);
    study_binary_functor<multiplication<scalar>>(vals);
    study_binary_functor<division<scalar>>(vals);
  }
};

template<typename scalar>
struct study_scalar_type_impl<scalar, false>
{
  static void run(const vector<scalar>& vals)
  {
    typedef typename other_floating_point_type<scalar>::type other_fp_type;
    study_unary_functor<conversion<scalar, other_fp_type>>(vals);
    study_unary_functor<conversion<scalar, int8_t>>(vals);
    study_unary_functor<conversion<scalar, int16_t>>(vals);
    study_unary_functor<conversion<scalar, int32_t>>(vals);
    study_unary_functor<conversion<scalar, int64_t>>(vals);

    study_unary_functor<exponential<scalar>>(vals);
    study_unary_functor<logarithm<scalar>>(vals);
    study_unary_functor<cosine<scalar>>(vals);
    study_unary_functor<arctangent<scalar>>(vals);

    study_binary_functor<addition<scalar>>(vals);
    study_binary_functor<subtraction<scalar>>(vals);
    study_binary_functor<multiplication<scalar>>(vals);
    study_binary_functor<division<scalar>>(vals);
  }
};

template<typename scalar>
void study_scalar_type(const vector<scalar>& vals)
{
  study_scalar_type_impl<scalar>::run(vals);
}

int main(int argc, char*argv[])
{
#ifdef ARM_DISABLE_FZ
  __asm__ volatile("vmrs r0, fpscr\n"
                    "bic r0, $(1 << 24)\n"
                    "vmsr fpscr, r0" : : : "r0");
#endif

#ifdef ARM_ENABLE_FZ
  __asm__ volatile("vmrs r0, fpscr\n"
                    "orr r0, $(1 << 24)\n"
                    "vmsr fpscr, r0" : : : "r0");
#endif

#ifdef X86_ENABLE_DAZ
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  assert(_MM_GET_DENORMALS_ZERO_MODE() & _MM_DENORMALS_ZERO_ON);
#endif

#ifdef X86_ENABLE_FTZ
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  assert(_MM_GET_FLUSH_ZERO_MODE() & _MM_FLUSH_ZERO_ON);
#endif

  cout.precision(lowprecision);

  vector<float> floatvals;
  floatvals.push_back(0.f);
  floatvals.push_back(numeric_limits<float>::denorm_min());
  floatvals.push_back(1e-44f);
  floatvals.push_back(-1e-44f);
  floatvals.push_back(1e-40f);
  floatvals.push_back(-1e-40f);
  floatvals.push_back(1e-36f);
  floatvals.push_back(1e-32f);
  floatvals.push_back(1.f / 33331.f);
  floatvals.push_back(1.f / 256.f);
  floatvals.push_back(-1.f / 256.f);
  floatvals.push_back(1.f / 31.f);
  floatvals.push_back(0.5f);
  floatvals.push_back(1.f);
  floatvals.push_back(-1.f);
  floatvals.push_back(2.f);
  floatvals.push_back(255.f);
  floatvals.push_back(-255.f);
  floatvals.push_back(256.f);
  floatvals.push_back(-256.f);
  floatvals.push_back(65535.f);
  floatvals.push_back(65536.f);
  floatvals.push_back(16777215.f);
  floatvals.push_back(-16777215.f);
  floatvals.push_back(16777216.f);
  floatvals.push_back(-16777216.f);
  floatvals.push_back(1e16f);
  floatvals.push_back(-1e16f);
  floatvals.push_back(1e32f);
  floatvals.push_back(numeric_limits<float>::infinity());
  floatvals.push_back(-numeric_limits<float>::infinity());
  floatvals.push_back(numeric_limits<float>::quiet_NaN());
  floatvals.push_back(numeric_limits<float>::signaling_NaN());

  vector<double> doublevals;
  doublevals.push_back(0.);
  doublevals.push_back(numeric_limits<double>::denorm_min());
  doublevals.push_back(1e-320);
  doublevals.push_back(-1e-320);
  doublevals.push_back(1e-310);
  doublevals.push_back(-1e-310);
  doublevals.push_back(1e-306);
  doublevals.push_back(1e-306);
  doublevals.push_back(1. / 33331.);
  doublevals.push_back(1. / 256.);
  doublevals.push_back(-1. / 256.);
  doublevals.push_back(1. / 31.);
  doublevals.push_back(0.5);
  doublevals.push_back(1.);
  doublevals.push_back(-1.);
  doublevals.push_back(2.);
  doublevals.push_back(255.);
  doublevals.push_back(-255.);
  doublevals.push_back(256.);
  doublevals.push_back(-256.);
  doublevals.push_back(65535.);
  doublevals.push_back(65536.);
  doublevals.push_back(16777215.);
  doublevals.push_back(-16777215.);
  doublevals.push_back(16777216.);
  doublevals.push_back(-16777216.);
  doublevals.push_back(1e16);
  doublevals.push_back(-1e16);
  doublevals.push_back(1e300);
  doublevals.push_back(numeric_limits<double>::infinity());
  doublevals.push_back(-numeric_limits<double>::infinity());
  doublevals.push_back(numeric_limits<double>::quiet_NaN());
  doublevals.push_back(numeric_limits<double>::signaling_NaN());

  vector<int8_t> int8vals;
  int8vals.push_back(numeric_limits<int8_t>::min());
  int8vals.push_back(numeric_limits<int8_t>::min() + 11);
  int8vals.push_back(-11);
  int8vals.push_back(-1);
  int8vals.push_back(1);
  int8vals.push_back(1);
  int8vals.push_back(11);
  int8vals.push_back(numeric_limits<int8_t>::max() - 11);
  int8vals.push_back(numeric_limits<int8_t>::max());

  vector<int16_t> int16vals;
  int16vals.push_back(numeric_limits<int16_t>::min());
  int16vals.push_back(numeric_limits<int16_t>::min() + 11);
  int16vals.push_back(-1111);
  int16vals.push_back(-11);
  int16vals.push_back(-1);
  int16vals.push_back(1);
  int16vals.push_back(1);
  int16vals.push_back(11);
  int16vals.push_back(1111);
  int16vals.push_back(numeric_limits<int16_t>::max() - 11);
  int16vals.push_back(numeric_limits<int16_t>::max());

  vector<int32_t> int32vals;
  int32vals.push_back(numeric_limits<int32_t>::min());
  int32vals.push_back(numeric_limits<int32_t>::min() + 11);
  int32vals.push_back(-1111111);
  int32vals.push_back(-11);
  int32vals.push_back(-1);
  int32vals.push_back(1);
  int32vals.push_back(1);
  int32vals.push_back(11);
  int32vals.push_back(1111111);
  int32vals.push_back(numeric_limits<int32_t>::max() - 11);
  int32vals.push_back(numeric_limits<int32_t>::max());

  vector<int64_t> int64vals;
  int64vals.push_back(numeric_limits<int64_t>::min());
  int64vals.push_back(numeric_limits<int64_t>::min() + 11);
  int64vals.push_back(-1111111111111LL);
  int64vals.push_back(-1111111);
  int64vals.push_back(-11);
  int64vals.push_back(-1);
  int64vals.push_back(1);
  int64vals.push_back(1);
  int64vals.push_back(11);
  int64vals.push_back(1111111);
  int64vals.push_back(1111111111111LL);
  int64vals.push_back(numeric_limits<int64_t>::max() - 11);
  int64vals.push_back(numeric_limits<int64_t>::max());

  study_scalar_type<float>(floatvals);
  study_scalar_type<double>(doublevals);
  study_scalar_type<int8_t>(int8vals);
  study_scalar_type<int16_t>(int16vals);
  study_scalar_type<int32_t>(int32vals);
  study_scalar_type<int64_t>(int64vals);

  if (alerts.size()) {
    cout << "There are " << alerts.size()
         << " alerts about exploitable timing differences:" << endl;
    for (vector<string>::const_iterator it = alerts.begin();
        it != alerts.end();
        ++it)
    {
      cout << "  " << *it << endl;
    }
  } else {
    cout << "No alerts were reported. Good." << endl;
  }
}
