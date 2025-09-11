gol_rs v0.1.0 (D:\code\Apps\gol_rs)
├── image v0.25.8
│   ├── bytemuck v1.23.2
│   ├── byteorder-lite v0.1.0
│   ├── color_quant v1.1.0
│   ├── exr v1.73.0
│   │   ├── bit_field v0.10.3
│   │   ├── half v2.6.0
│   │   │   └── cfg-if v1.0.3
│   │   ├── lebe v0.5.3
│   │   ├── miniz_oxide v0.8.9
│   │   │   ├── adler2 v2.0.1
│   │   │   └── simd-adler32 v0.3.7
│   │   ├── rayon-core v1.13.0
│   │   │   ├── crossbeam-deque v0.8.6
│   │   │   │   ├── crossbeam-epoch v0.9.18
│   │   │   │   │   └── crossbeam-utils v0.8.21
│   │   │   │   └── crossbeam-utils v0.8.21
│   │   │   └── crossbeam-utils v0.8.21
│   │   ├── smallvec v1.15.1
│   │   └── zune-inflate v0.2.54
│   │       └── simd-adler32 v0.3.7
│   ├── gif v0.13.3
│   │   ├── color_quant v1.1.0
│   │   └── weezl v0.1.10
│   ├── image-webp v0.2.4
│   │   ├── byteorder-lite v0.1.0
│   │   └── quick-error v2.0.1
│   ├── moxcms v0.7.5
│   │   ├── num-traits v0.2.19
│   │   │   [build-dependencies]
│   │   │   └── autocfg v1.5.0
│   │   └── pxfm v0.1.23
│   │       └── num-traits v0.2.19 (*)
│   ├── num-traits v0.2.19 (*)
│   ├── png v0.18.0
│   │   ├── bitflags v2.9.4
│   │   ├── crc32fast v1.5.0
│   │   │   └── cfg-if v1.0.3
│   │   ├── fdeflate v0.3.7
│   │   │   └── simd-adler32 v0.3.7
│   │   ├── flate2 v1.1.2
│   │   │   ├── crc32fast v1.5.0 (*)
│   │   │   └── miniz_oxide v0.8.9 (*)
│   │   └── miniz_oxide v0.8.9 (*)
│   ├── qoi v0.4.1
│   │   └── bytemuck v1.23.2
│   ├── ravif v0.11.20
│   │   ├── avif-serialize v0.8.6
│   │   │   └── arrayvec v0.7.6
│   │   ├── imgref v1.11.0
│   │   ├── loop9 v0.1.5
│   │   │   └── imgref v1.11.0
│   │   ├── quick-error v2.0.1
│   │   ├── rav1e v0.7.1
│   │   │   ├── arg_enum_proc_macro v0.3.4 (proc-macro)
│   │   │   │   ├── proc-macro2 v1.0.101
│   │   │   │   │   └── unicode-ident v1.0.19
│   │   │   │   ├── quote v1.0.40
│   │   │   │   │   └── proc-macro2 v1.0.101 (*)
│   │   │   │   └── syn v2.0.106
│   │   │   │       ├── proc-macro2 v1.0.101 (*)
│   │   │   │       ├── quote v1.0.40 (*)
│   │   │   │       └── unicode-ident v1.0.19
│   │   │   ├── arrayvec v0.7.6
│   │   │   ├── av1-grain v0.2.4
│   │   │   │   ├── anyhow v1.0.99
│   │   │   │   ├── arrayvec v0.7.6
│   │   │   │   ├── log v0.4.28
│   │   │   │   ├── nom v7.1.3
│   │   │   │   │   ├── memchr v2.7.5
│   │   │   │   │   └── minimal-lexical v0.2.1
│   │   │   │   ├── num-rational v0.4.2
│   │   │   │   │   ├── num-bigint v0.4.6
│   │   │   │   │   │   ├── num-integer v0.1.46
│   │   │   │   │   │   │   └── num-traits v0.2.19 (*)
│   │   │   │   │   │   └── num-traits v0.2.19 (*)
│   │   │   │   │   ├── num-integer v0.1.46 (*)
│   │   │   │   │   └── num-traits v0.2.19 (*)
│   │   │   │   └── v_frame v0.3.9
│   │   │   │       ├── aligned-vec v0.6.4
│   │   │   │       │   └── equator v0.4.2
│   │   │   │       │       └── equator-macro v0.4.2 (proc-macro)
│   │   │   │       │           ├── proc-macro2 v1.0.101 (*)
│   │   │   │       │           ├── quote v1.0.40 (*)
│   │   │   │       │           └── syn v2.0.106 (*)
│   │   │   │       └── num-traits v0.2.19 (*)
│   │   │   ├── bitstream-io v2.6.0
│   │   │   ├── cfg-if v1.0.3
│   │   │   ├── itertools v0.12.1
│   │   │   │   └── either v1.15.0
│   │   │   ├── libc v0.2.175
│   │   │   ├── log v0.4.28
│   │   │   ├── maybe-rayon v0.1.1
│   │   │   │   ├── cfg-if v1.0.3
│   │   │   │   └── rayon v1.11.0
│   │   │   │       ├── either v1.15.0
│   │   │   │       └── rayon-core v1.13.0 (*)
│   │   │   ├── new_debug_unreachable v1.0.6
│   │   │   ├── noop_proc_macro v0.3.0 (proc-macro)
│   │   │   ├── num-derive v0.4.2 (proc-macro)
│   │   │   │   ├── proc-macro2 v1.0.101 (*)
│   │   │   │   ├── quote v1.0.40 (*)
│   │   │   │   └── syn v2.0.106 (*)
│   │   │   ├── num-traits v0.2.19 (*)
│   │   │   ├── once_cell v1.21.3
│   │   │   ├── paste v1.0.15 (proc-macro)
│   │   │   ├── profiling v1.0.17
│   │   │   │   └── profiling-procmacros v1.0.17 (proc-macro)
│   │   │   │       ├── quote v1.0.40 (*)
│   │   │   │       └── syn v2.0.106 (*)
│   │   │   ├── simd_helpers v0.1.0 (proc-macro)
│   │   │   │   └── quote v1.0.40 (*)
│   │   │   ├── thiserror v1.0.69
│   │   │   │   └── thiserror-impl v1.0.69 (proc-macro)
│   │   │   │       ├── proc-macro2 v1.0.101 (*)
│   │   │   │       ├── quote v1.0.40 (*)
│   │   │   │       └── syn v2.0.106 (*)
│   │   │   └── v_frame v0.3.9 (*)
│   │   │   [build-dependencies]
│   │   │   └── built v0.7.7
│   │   ├── rayon v1.11.0 (*)
│   │   └── rgb v0.8.52
│   ├── rayon v1.11.0 (*)
│   ├── rgb v0.8.52
│   ├── tiff v0.10.3
│   │   ├── fax v0.2.6
│   │   │   └── fax_derive v0.2.0 (proc-macro)
│   │   │       ├── proc-macro2 v1.0.101 (*)
│   │   │       ├── quote v1.0.40 (*)
│   │   │       └── syn v2.0.106 (*)
│   │   ├── flate2 v1.1.2 (*)
│   │   ├── half v2.6.0 (*)
│   │   ├── quick-error v2.0.1
│   │   ├── weezl v0.1.10
│   │   └── zune-jpeg v0.4.21
│   │       └── zune-core v0.4.12
│   ├── zune-core v0.4.12
│   └── zune-jpeg v0.4.21 (*)
├── rand v0.9.2
│   ├── rand_chacha v0.9.0
│   │   ├── ppv-lite86 v0.2.21
│   │   │   └── zerocopy v0.8.27
│   │   └── rand_core v0.9.3
│   │       └── getrandom v0.3.3
│   │           └── cfg-if v1.0.3
│   └── rand_core v0.9.3 (*)
└── rayon v1.11.0 (*)