PS D:\code\Apps\gol_rs> $env:RUSTFLAGS="-Awarnings"; cargo run --release
   Compiling gol_rs v0.1.0 (D:\code\Apps\gol_rs)

thread 'rustc' (33548) panicked at compiler\rustc_errors\src\lib.rs:726:17:
`trimmed_def_paths` called, diagnostics were expected but none were emitted. Use `with_no_trimmed_paths` for debugging. Backtraces are currently disabled: set `RUST_BACKTRACE=1` and re-run to see where it happened.
stack backtrace:
   0:     0x7ff878683dc2 - std::backtrace_rs::backtrace::win64::trace
                               at /rustc/7ad23f43a225546c095123de52cc07d8719f8e2b/library\std\src\..\..\backtrace\src\backtrace\win64.rs:85
   1:     0x7ff878683dc2 - std::backtrace_rs::backtrace::trace_unsynchronized
                               at /rustc/7ad23f43a225546c095123de52cc07d8719f8e2b/library\std\src\..\..\backtrace\src\backtrace\mod.rs:66
   2:     0x7ff878683dc2 - std::sys::backtrace::_print_fmt
                               at /rustc/7ad23f43a225546c095123de52cc07d8719f8e2b/library\std\src\sys\backtrace.rs:66
   3:     0x7ff878683dc2 - std::sys::backtrace::impl$0::print::impl$0::fmt
                               at /rustc/7ad23f43a225546c095123de52cc07d8719f8e2b/library\std\src\sys\backtrace.rs:39
   4:     0x7ff87869950a - core::fmt::rt::Argument::fmt
                               at /rustc/7ad23f43a225546c095123de52cc07d8719f8e2b/library\core\src\fmt\rt.rs:173
   5:     0x7ff87869950a - core::fmt::write
                               at /rustc/7ad23f43a225546c095123de52cc07d8719f8e2b/library\core\src\fmt\mod.rs:1468
   6:     0x7ff87864a8de - std::io::default_write_fmt
                               at /rustc/7ad23f43a225546c095123de52cc07d8719f8e2b/library\std\src\io\mod.rs:639
   7:     0x7ff87864a8de - std::io::Write::write_fmt<std::sys::stdio::windows::Stderr>
                               at /rustc/7ad23f43a225546c095123de52cc07d8719f8e2b/library\std\src\io\mod.rs:1954
   8:     0x7ff878663045 - std::sys::backtrace::BacktraceLock::print
                               at /rustc/7ad23f43a225546c095123de52cc07d8719f8e2b/library\std\src\sys\backtrace.rs:42
   9:     0x7ff87866a9a9 - std::panicking::default_hook::closure$0
                               at /rustc/7ad23f43a225546c095123de52cc07d8719f8e2b/library\std\src\panicking.rs:301
  10:     0x7ff87866a798 - std::panicking::default_hook
                               at /rustc/7ad23f43a225546c095123de52cc07d8719f8e2b/library\std\src\panicking.rs:328
  11:     0x7ff879e56146 - core[9201c249597990c3]::slice::sort::unstable::heapsort::heapsort::<((rustc_lint_defs[c9e85401c0f82309]::Level, &str), usize), <((rustc_lint_defs[c9e85401c0f82309]::Level, &str), usize) as core[9201c249597990c3]::cmp::PartialOrd>::lt>
  12:     0x7ff87866b32a - std::panicking::panic_with_hook
                               at /rustc/7ad23f43a225546c095123de52cc07d8719f8e2b/library\std\src\panicking.rs:842
  13:     0x7ff87866b0b9 - std::panicking::panic_handler::closure$0
                               at /rustc/7ad23f43a225546c095123de52cc07d8719f8e2b/library\std\src\panicking.rs:707
  14:     0x7ff87866324f - std::sys::backtrace::__rust_end_short_backtrace<std::panicking::panic_handler::closure_env$0,never$>
                               at /rustc/7ad23f43a225546c095123de52cc07d8719f8e2b/library\std\src\sys\backtrace.rs:174
  15:     0x7ff878644fbe - std::panicking::panic_handler
                               at /rustc/7ad23f43a225546c095123de52cc07d8719f8e2b/library\std\src\panicking.rs:698
  16:     0x7ff87c7a8481 - core::panicking::panic_fmt
                               at /rustc/7ad23f43a225546c095123de52cc07d8719f8e2b/library\core\src\panicking.rs:75
  17:     0x7ff8785326e0 - <rustc_errors[6219e5a8876ecfbf]::DiagCtxtInner as core[9201c249597990c3]::ops::drop::Drop>::drop
  18:     0x7ff875c08fa0 - llvm::function_ref<void __cdecl(llvm::Value const * __ptr64)>::callback_fn<`llvm::Value::stripInBoundsOffsets'::`1'::<lambda_1_1> >
  19:     0x7ff875c0989a - llvm::function_ref<void __cdecl(llvm::Value const * __ptr64)>::callback_fn<`llvm::Value::stripInBoundsOffsets'::`1'::<lambda_1_1> >
  20:     0x7ff875c2b6b1 - llvm::function_ref<void __cdecl(llvm::Value const * __ptr64)>::callback_fn<`llvm::Value::stripInBoundsOffsets'::`1'::<lambda_1_1> >
  21:     0x7ff875c37569 - std[c16ba45f833980ee]::sys::backtrace::__rust_begin_short_backtrace::<<std[c16ba45f833980ee]::thread::Builder>::spawn_unchecked_<ctrlc[cb92e8f8cd125a41]::set_handler_inner<rustc_driver_impl[7ac85c7c565a839e]::install_ctrlc_handler::{closure#0}>::{closure#0}, ()>::{closure#1}::{closure#0}::{closure#0}, ()>
  22:     0x7ff875c3298f - RINvNtNtCsgBzviSfxLkM_3std3sys9backtrace28___rust_begin_short_backtraceNCNCINvNtCs4h6t7xVMy5d_15rustc_interface4util26run_in_thread_with_globalsNCINvB1e_31run_in_thread_pool_with_globalsNCINvNtB1g_9interface12run_compileruNCNvCsaxzi69B8nfK_17rustc_driver_i
  23:     0x7ff875c40dbd - std[c16ba45f833980ee]::sys::backtrace::__rust_begin_short_backtrace::<<std[c16ba45f833980ee]::thread::Builder>::spawn_unchecked_<ctrlc[cb92e8f8cd125a41]::set_handler_inner<rustc_driver_impl[7ac85c7c565a839e]::install_ctrlc_handler::{closure#0}>::{closure#0}, ()>::{closure#1}::{closure#0}::{closure#0}, ()>
  24:     0x7ff878658f2d - alloc::boxed::impl$29::call_once
                               at /rustc/7ad23f43a225546c095123de52cc07d8719f8e2b/library\alloc\src\boxed.rs:1985
  25:     0x7ff878658f2d - alloc::boxed::impl$29::call_once
                               at /rustc/7ad23f43a225546c095123de52cc07d8719f8e2b/library\alloc\src\boxed.rs:1985
  26:     0x7ff878658f2d - std::sys::pal::windows::thread::impl$0::new::thread_start
                               at /rustc/7ad23f43a225546c095123de52cc07d8719f8e2b/library\std\src\sys\pal\windows\thread.rs:60
  27:     0x7ff9a781e8d7 - BaseThreadInitThunk
  28:     0x7ff9a819c34c - RtlUserThreadStart

error: the compiler unexpectedly panicked. this is a bug.

note: we would appreciate a bug report: https://github.com/rust-lang/rust/issues/new?labels=C-bug%2C+I-ICE%2C+T-compiler&template=ice.md

note: please make sure that you have updated to the latest nightly

note: please attach the file at `D:\code\Apps\gol_rs\rustc-ice-2025-09-10T23_22_12-18140.txt` to your bug report

note: compiler flags: --crate-type bin -C opt-level=3 -C lto=thin -C codegen-units=1 -C strip=symbols

note: some of the compiler flags provided by cargo are hidden

query stack during panic:
end of query stack
error: could not compile `gol_rs` (bin "gol_rs")

Caused by:
  process didn't exit successfully: `C:\Users\Gavin\.rustup\toolchains\nightly-x86_64-pc-windows-msvc\bin\rustc.exe --crate-name gol_rs --edition=2024 src\main.rs --error-format=json --json=diagnostic-rendered-ansi,artifacts,future-incompat --diagnostic-width=248 --crate-type bin --emit=dep-info,link -C opt-level=3 -C lto=thin -C codegen-units=1 --check-cfg cfg(docsrs,test) --check-cfg "cfg(feature, values())" -C metadata=7d3c2265bd36ddf3 --out-dir D:\code\Apps\gol_rs\target\release\deps -C strip=symbols -L dependency=D:\code\Apps\gol_rs\target\release\deps --extern image=D:\code\Apps\gol_rs\target\release\deps\libimage-57cb4050bfcf6ef1.rlib --extern rand=D:\code\Apps\gol_rs\target\release\deps\librand-c555bb54afe7816e.rlib --extern rayon=D:\code\Apps\gol_rs\target\release\deps\librayon-976b8e5a512dd73a.rlib -Awarnings` (exit code: 101)