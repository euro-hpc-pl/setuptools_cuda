def customize_compiler_for_nvcc(compiler):
    compiler.src_extensions.append(".cu")
    default_compiler_so = compiler.compiler_so
    old_compile = compiler._compile
    auto_depends = getattr(compiler, "_auto_depends", False)

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if ext == ".cu" and not compiler.has_nvcc:
            raise RuntimeError("Compiling CUDA extensions requires nvcc.")
        if ext == ".cu":
            postargs = [
                "-Xcompiler",
                "-fPIC",
                "-lstdc++",
                "-shared",
            ]
            compiler.set_executable("compiler_so", "nvcc")
            compiler._auto_depends = False
        else:
            compiler.set_executable("compiler_so", default_compiler_so)
            postargs = extra_postargs

        old_compile(obj, src, ext, cc_args, postargs, pp_opts)
        compiler.compiler_so = default_compiler_so
        setattr(compiler, "_auto_depends", auto_depends)

    compiler._compile = _compile
