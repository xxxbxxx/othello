const std = @import("std");
const assert = std.debug.assert;

pub const Coord = [2]i8; //@Vector(2, i8); ..selfhosted
fn add(a: Coord, b: Coord) Coord {
    return .{ a[0] + b[0], a[1] + b[1] };
}
fn inBounds(p: Coord, comptime min: Coord, comptime max: Coord) bool {
    // @reduce(.And, @min(@max(p, V2i{ 0, 0 }), V2i{ 7, 7 }) == p))
    return @min(@max(p[0], min[0]), max[0]) == p[0] // x
    and @min(@max(p[1], min[1]), max[1]) == p[1]; // y
}

pub fn bitmask(x: anytype, y: anytype) u64 {
    assert(x >= 0 and y >= 0 and x < 8 and y < 8);
    return @as(u64, 1) << @intCast(y * 8 + x);
}

pub const Color = enum {
    empty,
    black,
    white,
    pub fn next(c: Color) Color {
        return switch (c) {
            .empty => .empty,
            .black => .white,
            .white => .black,
        };
    }
};
pub const Board = struct {
    occupied: u64,
    color: u64,

    pub fn get(b: @This(), x: anytype, y: anytype) Color {
        const bit: u64 = bitmask(x, y);
        if ((b.occupied & bit) == 0) return .empty;
        return if ((b.color & bit) == 0) .black else .white;
    }
    pub fn setBitmask(b: *@This(), bit: u64, c: Color) void {
        switch (c) {
            .empty => b.occupied &= ~bit,
            .black => {
                b.occupied |= bit;
                b.color &= ~bit;
            },
            .white => {
                b.occupied |= bit;
                b.color |= bit;
            },
        }
    }

    pub fn set(b: *@This(), x: anytype, y: anytype, c: Color) void {
        const bit: u64 = bitmask(x, y);
        setBitmask(b, bit, c);
    }
};

pub const init_board: Board = init: {
    var b: Board = .{ .color = 0, .occupied = 0 };
    b.set(3, 3, .white);
    b.set(4, 4, .white);
    b.set(3, 4, .black);
    b.set(4, 3, .black);
    break :init b;
};

const dirsteps: [8 * 8][8][]const u64 = blk: {
    @setEvalBranchQuota(10000);

    const compass_dirs: []const Coord = &.{
        .{ 1, 0 }, .{ -1, 0 }, .{ 0, 1 },  .{ 0, -1 },
        .{ 1, 1 }, .{ -1, 1 }, .{ 1, -1 }, .{ -1, -1 },
    };
    var allsteps: [8 * 8][8][]const u64 = undefined;
    for (0..8) |y| {
        for (0..8) |x| {
            nextdir: for (compass_dirs, 0..) |d, idir| {
                var steps: []const u64 = &.{};
                var p: Coord = .{ @intCast(x), @intCast(y) };
                while (true) {
                    p = add(p, d);
                    if (!inBounds(p, Coord{ 0, 0 }, Coord{ 7, 7 })) {
                        allsteps[y * 8 + x][idir] = steps;
                        continue :nextdir;
                    }
                    steps = steps ++ .{bitmask(p[0], p[1])};
                }
            }
        }
    }
    break :blk allsteps;
};

pub fn computeValidSquares(b: Board, col: Color) u64 {
    assert(col != .empty);
    var valid: u64 = 0;
    for (0..8) |y| {
        for (0..8) |x| {
            if (b.get(x, y) != .empty) continue;

            const ok: bool = loop: for (&dirsteps[y * 8 + x]) |steps| {
                var has_oppo = false;
                for (steps) |bit| {
                    if ((b.occupied & bit) == 0) continue :loop;
                    const sq: Color = if ((b.color & bit) == 0) .black else .white;
                    if (sq == col and !has_oppo) continue :loop;
                    if (sq == col and has_oppo) break :loop true;
                    has_oppo = true;
                }
            } else false;

            if (ok) {
                valid |= bitmask(x, y);
            }
        }
    }
    return valid;
}

pub fn playAt(b: Board, p: Coord, col: Color) !Board {
    assert(col != .empty);
    const valid = computeValidSquares(b, col);
    const bit0 = bitmask(p[0], p[1]);
    if (valid & bit0 == 0) return error.invalid;
    var b1 = b;
    b1.setBitmask(bit0, col);

    loop: for (&dirsteps[@intCast(p[1] * 8 + p[0])]) |steps| {
        for (steps) |bit| {
            if ((b.occupied & bit) == 0) continue :loop;
            const sq: Color = if ((b.color & bit) == 0) .black else .white;
            if (sq == col) break;
        } else continue :loop;

        for (steps) |bit| {
            assert((b.occupied & bit) != 0);
            const sq: Color = if ((b.color & bit) == 0) .black else .white;
            if (sq == col) continue :loop;
            b1.setBitmask(bit, col);
        }
    }
    return b1;
}

pub fn computeScore(b: Board) struct { whites: u32, blacks: u32 } {
    var nbw: u32 = 0;
    var nbb: u32 = 0;
    var bit: u64 = 1;
    for (0..64) |_| {
        nbb += @intFromBool((b.occupied & bit != 0) and (b.color & bit == 0));
        nbw += @intFromBool((b.occupied & bit != 0) and (b.color & bit != 0));
        bit <<= 1;
    }
    return .{ .whites = nbw, .blacks = nbb };
}

// bon pas d'état persistant pour l'instant
pub const Engine = enum(i32) {
    none,
    random,
    greedy,
    one_step,
    two_steps,
    five_steps,
};

const Context = struct {
    dico: std.AutoHashMap(Board, i32),
};

pub fn computeBestMove(engine: Engine, b: Board, col: Color, alloc: std.mem.Allocator, random: std.Random) Coord {
    var ctx: Context = .{
        .dico = std.AutoHashMap(Board, i32).init(alloc),
    };
    defer ctx.dico.deinit();

    return switch (engine) {
        .none => unreachable,
        .random => computeRandomMove(b, col, random),
        .greedy => computeGreedyMove(b, col, random).?.coord,
        .one_step => computeStepBestMove(b, col, random, &ctx, 1).?.coord,
        .two_steps => computeStepBestMove(b, col, random, &ctx, 2).?.coord,
        .five_steps => computeStepBestMove(b, col, random, &ctx, 5).?.coord,
    };
}

fn computeRandomMove(b: Board, col: Color, random: std.Random) Coord {
    const valids = computeValidSquares(b, col);
    while (true) {
        const index = random.intRangeAtMost(u6, 0, 63);
        const bit = @as(u64, 1) << index;
        if (valids & bit == 0) continue;
        return .{ index % 8, index / 8 };
    }
}

fn evaluation(b: Board, col: Color) i32 {
    const score = computeScore(b);
    const nb0: i64 = if (col == .white) score.whites else score.blacks;
    const nb1: i64 = if (col == .black) score.whites else score.blacks;
    const delta: i32 = @intCast(nb0 - nb1);
    return delta;
}

fn computeGreedyMove(b: Board, col: Color, random: std.Random) ?struct { coord: Coord, score: i32 } {
    const valids = computeValidSquares(b, col);

    var best: ?Coord = null;
    var best_score: i32 = 0;
    for (0..8) |y| {
        for (0..8) |x| {
            const bit = bitmask(x, y);
            if (valids & bit == 0) continue;

            const coord: Coord = .{ @intCast(x), @intCast(y) };
            const after = playAt(b, coord, col) catch unreachable;
            const delta = evaluation(after, col);
            if (best_score < delta or best == null) {
                best = coord;
                best_score = delta;
            } else if (best_score == delta and random.boolean()) {
                best = coord;
                best_score = delta;
            }
        }
    }
    return if (best) |coord| .{ .coord = coord, .score = best_score } else null;
}

fn computeStepBestMove(b: Board, col: Color, random: std.Random, ctx: *Context, lookahead: u32) ?struct { coord: Coord, score: i32 } {
    const valids = computeValidSquares(b, col);

    var best: ?Coord = null;
    var best_score: i32 = 0;
    for (0..8) |y| {
        for (0..8) |x| {
            const bit = bitmask(x, y);
            if (valids & bit == 0) continue;

            const coord: Coord = .{ @intCast(x), @intCast(y) };
            const after = playAt(b, coord, col) catch unreachable;

            const expected: i32 = score: {
                if (ctx.dico.get(after)) |v| break :score v;
                switch (lookahead) {
                    0 => {},
                    1 => {
                        if (computeGreedyMove(after, col.next(), random)) |res|
                            break :score -res.score;
                    },
                    else => {
                        if (computeStepBestMove(after, col.next(), random, ctx, lookahead - 1)) |res|
                            break :score -res.score;
                    },
                }
                break :score evaluation(after, col);
            };
            ctx.dico.put(after, expected) catch unreachable;
            if (best_score < expected or best == null) {
                best = coord;
                best_score = expected;
            } else if (best_score == expected and random.boolean()) {
                best = coord;
                best_score = expected;
            }
        }
    }
    return if (best) |coord| .{ .coord = coord, .score = best_score } else null;
}
