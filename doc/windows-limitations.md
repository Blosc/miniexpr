# Windows Limitations

## Complex Types

Complex types are not supported on Windows. The C99 complex ABI is not stable across MSVC and clang-cl, which leads to incorrect results when passing complex values between compiled code and the runtime. To avoid silent corruption, `me_compile()` rejects any expression that involves complex variables or complex outputs on Windows.

If you need complex arithmetic on Windows, use a custom pair type (real + imag) and handle complex math outside of miniexpr.
