const std = @import("std");
const assert = std.debug.assert;

pub const Coord = [2]i8; //@Vector(2, i8); ..selfhosted
fn add(a: Coord, b: Coord) Coord {
    return .{ a[0] + b[0], a[1] + b[1] };
}
fn inBounds(p: Coord, min: Coord, max: Coord) bool {
    // @reduce(.And, @min(@max(p, V2i{ 0, 0 }), V2i{ 7, 7 }) == p))
    return p[0] >= min[0] and p[0] <= max[0] and p[1] >= min[1] and p[1] <= max[1];
}

pub fn bitmask(x: anytype, y: anytype) u64 {
    assert(x >= 0 and y >= 0 and x < 8 and y < 8);
    return @as(u64, 1) << @intCast(y * 8 + x);
}

pub const Color = enum { empty, black, white };
pub const Board = struct {
    occupied: u64,
    color: u64,

    pub fn get(b: @This(), x: anytype, y: anytype) Color {
        const bit: u64 = bitmask(x, y);
        if ((b.occupied & bit) == 0) return .empty;
        return if ((b.color & bit) == 0) .black else .white;
    }
    pub fn set(b: *@This(), x: anytype, y: anytype, c: Color) void {
        const bit: u64 = bitmask(x, y);
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
};

pub const init_board: Board = init: {
    var b: Board = .{ .color = 0, .occupied = 0 };
    b.set(3, 3, .white);
    b.set(4, 4, .white);
    b.set(3, 4, .black);
    b.set(4, 3, .black);
    break :init b;
};

const compass_dirs: []const Coord = &.{
    .{ 1, 0 }, .{ -1, 0 }, .{ 0, 1 },  .{ 0, -1 },
    .{ 1, 1 }, .{ -1, 1 }, .{ 1, -1 }, .{ -1, -1 },
};

pub fn computeValidSquares(b: Board, col: Color) u64 {
    assert(col != .empty);
    var valid: u64 = 0;
    for (0..8) |y| {
        for (0..8) |x| {
            if (b.get(x, y) != .empty) continue;

            const ok: bool = loop: for (compass_dirs) |d| {
                var p: Coord = .{ @intCast(x), @intCast(y) };
                var has_oppo = false;
                while (true) {
                    p = add(p, d);
                    if (!inBounds(p, Coord{ 0, 0 }, Coord{ 7, 7 })) continue :loop;
                    const sq = b.get(p[0], p[1]);
                    if (sq == .empty) continue :loop;
                    if (sq == col and !has_oppo) continue :loop;
                    if (sq == col and has_oppo) break :loop true;
                    if (sq != col) has_oppo = true;
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
    const bit = bitmask(p[0], p[1]);
    if (valid & bit == 0) return error.invalid;
    var b1 = b;
    b1.set(p[0], p[1], col);

    loop: for (compass_dirs) |d| {
        var p1 = p;
        while (true) {
            p1 = add(p1, d);
            if (!inBounds(p1, Coord{ 0, 0 }, Coord{ 7, 7 })) continue :loop;
            const sq = b.get(p1[0], p1[1]);
            if (sq == .empty) continue :loop;
            if (sq == col) break;
        }

        p1 = p;
        while (true) {
            p1 = add(p1, d);
            assert(inBounds(p1, Coord{ 0, 0 }, Coord{ 7, 7 }));
            const sq = b.get(p1[0], p1[1]);
            assert(sq != .empty);
            if (sq == col) continue :loop;
            b1.set(p1[0], p1[1], col);
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
};

pub fn computeBestMove(engine: Engine, b: Board, col: Color, alloc: std.mem.Allocator, random: std.Random) Coord {
    _ = alloc;
    return switch (engine) {
        .none => unreachable,
        .random => computeRandomMove(b, col, random),
        .greedy => computeGreedyMove(b, col, random),
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

fn computeGreedyMove(b: Board, col: Color, random: std.Random) Coord {
    const valids = computeValidSquares(b, col);

    var best: ?Coord = null;
    var best_score: i32 = 0;
    for (0..8) |y| {
        for (0..8) |x| {
            const bit = bitmask(x, y);
            if (valids & bit == 0) continue;

            const coord: Coord = .{ @intCast(x), @intCast(y) };
            const after = playAt(b, coord, col) catch unreachable;
            const score = computeScore(after);
            const nb0: i64 = if (col == .white) score.whites else score.blacks;
            const nb1: i64 = if (col == .black) score.whites else score.blacks;
            const delta: i32 = @intCast(nb0 - nb1);
            if (best_score < delta or best == null) {
                best = coord;
                best_score = delta;
            } else if (best_score == delta and random.boolean()) {
                best = coord;
                best_score = delta;
            }
        }
    }
    return best.?;
}
