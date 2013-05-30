/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

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

// How many significant digits we really want to print for all those floats
const int lowprecision = 3;

// Minimum time in seconds that our payload needs to run to give a useful timing
const double minimum_measurement_time = 0.002;

// How many timings we run before we pick the median one
const size_t measurement_repetitions = 32;

// Thresholds to report / to generate an alert about an operation definitely taking a different
// amount of time depending on its operands. Expressed in standard deviations ("sigmas").
// A higher value means a fewer false positives. The following tables allows to
// pick a value for a given target false positives rate. From
// http://en.wikipedia.org/wiki/Standard_deviation#Rules_for_normally_distributed_data
//
//   Value  |  Chance of false positive
// ---------+-----------------------------
//   1.645  |  10 %
//   2.576  |  1 %
//   3.291  |  0.1 % (1e-3)
//   3.891  |  1e-4
//   4.417  |  1e-5
//   4.892  |  1e-6
//   5      |  LHC standard!
//   5.327  |  1e-7
//
const double significance_alert_threshold = 2.576;
const double significance_report_threshold = 1.645;

// Minimum number of timings to report for each operation even if nothing looks suspicious
const size_t minimum_reports = 8;

// Even if a timing is far enough from the median to be statistically significant,
// it also needs to be sufficiently different from the median to be exploitable.
// Otherwise we end up reporting a lot of near-median timings just because the stdev happens
// to be very low. In theory, these are exploitable, in practice, they aren't until
// the ratio of the timing to the median is substantial enough.
//
// The constant here is the threshold on max(timing, median) / min(timing, median).
// For example, a value of 1.2 would mean that the timing difference must be at least 20%
// to be worth an alert.
const double timing_ratio_alert_threshold = 1.2;

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
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
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
double median_timing(const functor& f)
{
  vector<double> timings;
  for (size_t i = 0; i < measurement_repetitions; i++) {
    timings.push_back(one_timing(f));
  }
  sort(timings.begin(), timings.end());
  return timings[timings.size()/2];
}

template<typename scalar>
scalar mean(const scalar* begin, const scalar* end)
{
  scalar accum = 0;
  for_each (begin, end, [&](const scalar x) {
      accum += x;
  });
  return accum / (end - begin);
}

template<typename scalar>
scalar stdev(const scalar* begin, const scalar* end, scalar mean)
{
  scalar accum = 0;
  for_each (begin, end, [&](const scalar x) {
      accum += (x - mean) * (x - mean);
  });

  return sqrt(accum / (end - begin - 1));
}

struct functor_result
{
  string name;
  double raw_values[measurement_repetitions];
  double mean;
  double stdev;
  double distance_to_overall_median_in_stdevs;

  bool operator< (const functor_result& other) const
  {
    return distance_to_overall_median_in_stdevs > other.distance_to_overall_median_in_stdevs;
  }
};

void analyze_functor_results(vector<functor_result>& functor_results)
{
  vector<double> all_raw_timings(functor_results.size() * measurement_repetitions);

  size_t raw_timing_index = 0;

  for (size_t i = 0; i < functor_results.size(); i++)
  {
    for (size_t repetition = 0; repetition < measurement_repetitions; repetition++)
    {
      all_raw_timings[raw_timing_index++] = functor_results[i].raw_values[repetition];
    }
  }
  sort(all_raw_timings.begin(), all_raw_timings.end());

  double overall_median = all_raw_timings[all_raw_timings.size() / 2];
  cout << "  overall median: " << overall_median << endl;

  size_t report;
  for (report = 0; report < functor_results.size(); report++)
  {
    functor_result& func_res = functor_results[report];
    func_res.mean = mean(func_res.raw_values, func_res.raw_values + measurement_repetitions);
    func_res.stdev = stdev(func_res.raw_values, func_res.raw_values + measurement_repetitions, func_res.mean);
    func_res.distance_to_overall_median_in_stdevs = std::abs(func_res.mean - overall_median) / func_res.stdev;
  }

  sort(functor_results.begin(), functor_results.end());

  for (report = 0; report < functor_results.size(); report++)
  {
    functor_result& func_res = functor_results[report];
    if (report >= minimum_reports && func_res.distance_to_overall_median_in_stdevs < significance_report_threshold) {
      break;
    }
    cout << "  " << func_res.name << " : mean time " << func_res.mean << " s, stdev " << func_res.stdev << " s, distance from overall median " << func_res.distance_to_overall_median_in_stdevs << " stdevs" << endl;
  }
  if (++report < functor_results.size()) {
    cout << "  the remaining " << (functor_results.size() - report) << " timings are within "
         <<  functor_results[report].distance_to_overall_median_in_stdevs << " sigma of the overall median" << endl;
  }

  if (functor_results[0].distance_to_overall_median_in_stdevs >= significance_alert_threshold) {
    double a = min(functor_results[0].mean, overall_median);
    double b = max(functor_results[0].mean, overall_median);
    if (b / a > timing_ratio_alert_threshold) {
      lowprecisionstringstream alert;
      size_t alert_number = alerts.size() + 1;
      alert << "  alert #" << alert_number
            << ": deviation of " << functor_results[0].distance_to_overall_median_in_stdevs
            << " sigma for " << functor_results[0].name;
      alerts.push_back(alert.str());
      cout << alerts.back() << endl;
    }
  }

  cout << endl;
}

template<typename functor, typename scalar>
void study_binary_functor(const vector<scalar>& scalars)
{
  cout << functor::generic_name() << endl;

  vector<functor_result> functor_results(scalars.size() * scalars.size());

  cout << "  timing";

  for (size_t repetition = 0; repetition < measurement_repetitions; repetition++)
  {
    cout << "." << flush;
    for (size_t ix = 0; ix < scalars.size(); ix++)
    {
      for (size_t iy = 0; iy < scalars.size(); iy++)
      {
        functor_result& func_res = functor_results[ix * scalars.size() + iy];
        functor f(scalars[ix], scalars[iy]);
        if (repetition == 0) {
          func_res.name = f.name();
        }
        double value = one_timing(f);
        func_res.raw_values[repetition] = value;
      }
    }
  }
  cout << endl;

  analyze_functor_results(functor_results);
}

template<typename functor, typename scalar>
void study_unary_functor(const vector<scalar>& scalars)
{
  cout << functor::generic_name() << endl;

  vector<functor_result> functor_results(scalars.size());

  cout << "  timing";

  for (size_t repetition = 0; repetition < measurement_repetitions; repetition++)
  {
    cout << "." << flush;
    for (size_t ix = 0; ix < scalars.size(); ix++)
    {
      functor_result& func_res = functor_results[ix];
      functor f(scalars[ix]);
      if (repetition == 0) {
        func_res.name = f.name();
      }
      double value = one_timing(f);
      func_res.raw_values[repetition] = value;
    }
  }
  cout << endl;

  analyze_functor_results(functor_results);
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

int main()
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
