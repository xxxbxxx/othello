const std = @import("std");
const rlz = @import("raylib-zig");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const raylib_dep = b.dependency("raylib-zig", .{
        .target = target,
        .optimize = .ReleaseFast,
        .shared = false,
        .linux_display_backend = .Wayland,
        .rmodels = false,
    });

    const raylib = raylib_dep.module("raylib");
    const raygui = raylib_dep.module("raygui");
    const raylib_artifact = raylib_dep.artifact("raylib");

    //web exports are completely separate
    if (target.query.os_tag == .emscripten) {
        const exe_lib = rlz.emcc.compileForEmscripten(b, "othello", "src/main.zig", target, optimize);

        exe_lib.linkLibrary(raylib_artifact);
        exe_lib.root_module.addImport("raylib", raylib);
        exe_lib.root_module.addImport("raygui", raygui);

        // Note that raylib itself is not actually added to the exe_lib output file, so it also needs to be linked with emscripten.
        const link_step = try rlz.emcc.linkWithEmscripten(b, &[_]*std.Build.Step.Compile{ exe_lib, raylib_artifact });

        b.getInstallStep().dependOn(&link_step.step);
        const run_step = try rlz.emcc.emscriptenRunStep(b);
        run_step.step.dependOn(&link_step.step);
        const run_option = b.step("run", "Run othello");
        run_option.dependOn(&run_step.step);
        return;
    }

    const exe = b.addExecutable(.{
        .name = "othello",
        .root_source_file = b.path("src/main.zig"),
        .optimize = optimize,
        .target = target,
        .use_llvm = (optimize != .Debug),
        .use_lld = (optimize != .Debug),
    });

    exe.linkLibrary(raylib_artifact);
    exe.root_module.addImport("raylib", raylib);
    exe.root_module.addImport("raygui", raygui);

    const run_cmd = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run othello");
    run_step.dependOn(&run_cmd.step);

    b.installArtifact(exe);

    const main_test = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .use_llvm = (optimize != .Debug),
        .use_lld = (optimize != .Debug),
    });
    const test_cmd = b.addRunArtifact(main_test);
    const test_step = b.step("test", "run tests");
    test_step.dependOn(&test_cmd.step);
}
