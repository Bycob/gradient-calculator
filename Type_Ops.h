#include "armadillo/armadillo"

namespace netn {

	template <typename L, typename R>
	struct add_t {
		typedef decltype((L)0 + (R)0) type;
	};

	template <> struct add_t<arma::mat, arma::mat> { typedef arma::mat type; };
	template <typename T> struct add_t<arma::mat, T> { typedef arma::mat type; };
	template <typename T> struct add_t<T, arma::mat> { typedef arma::mat type; };

	template <typename L, typename R>
	struct sub_t {
		typedef decltype((L)0 - (R)0) type;
	};

	template <> struct sub_t<arma::mat, arma::mat> { typedef arma::mat type; };
	template <typename T> struct sub_t<arma::mat, T> { typedef arma::mat type; };
	template <typename T> struct sub_t<T, arma::mat> { typedef arma::mat type; };

	template <typename L, typename R>
	struct mul_t {
		typedef decltype((L)0 * (R)0) type;
	};

	template <> struct mul_t<arma::mat, arma::mat> { typedef arma::mat type; };
	template <typename T> struct mul_t<arma::mat, T> { typedef arma::mat type; };
	template <typename T> struct mul_t<T, arma::mat> { typedef arma::mat type; };

	template <typename L, typename R>
	struct div_t {
		typedef decltype((L)0 / (R)0) type;
	};

	template <typename T> struct div_t<arma::mat, T> { typedef arma::mat type; };
}