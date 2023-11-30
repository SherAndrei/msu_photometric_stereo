#pragma once

#if !defined(NDEBUG)
#	define VERIFY(expr) assert(expr)
#	define VERIFY_RETURN(expr, ret_expr) assert(expr)
#	define VERIFY_LOG_L(level, expr, log_expr)  assert(expr)
#	define VERIFY_LOG(expr, log_expr) assert(expr)
#	define VERIFY_LOG_RETURN_L(level, expr, log_expr, ret_expr)  assert(expr)
#	define VERIFY_LOG_RETURN(expr, log_expr, ret_expr) assert(expr)
#else
#	define VERIFY(expr) static_cast<void>(expr)
#	define VERIFY_RETURN(expr, ret_expr) \
		do { \
			if (!static_cast<bool>(expr)) \
				return ret_expr; \
		} while(0)
#	define VERIFY_LOG_L(level, expr, log_expr) \
		do { \
			if (!static_cast<bool>(expr)) \
				l << log_expr; \
		} while(0)
#	define VERIFY_LOG(expr, log_expr) VERIFY_LOG_L(std::cerr, expr, log_expr)
#	define VERIFY_LOG_RETURN_L(l, expr, log_expr, ret_expr) \
		do { \
			if (!static_cast<bool>(expr)) { \
				l << log_expr; \
				return ret_expr; \
			} \
		} while(0)
#	define VERIFY_LOG_RETURN(expr, log_expr, ret_expr) VERIFY_LOG_RETURN_L(std::cerr, expr, log_expr, ret_expr)
#endif
