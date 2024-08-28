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
    white: u64,
    black: u64,

    pub fn get(b: @This(), x: anytype, y: anytype) Color {
        const bit: u64 = bitmask(x, y);
        if ((b.white & bit) != 0) return .white;
        if ((b.black & bit) != 0) return .black;
        return .empty;
    }

    pub fn setBitmask(b: *@This(), bit: u64, c: Color) void {
        switch (c) {
            .empty => {
                b.white &= ~bit;
                b.black &= ~bit;
            },
            .white => {
                b.white |= bit;
                b.black &= ~bit;
            },
            .black => {
                b.black |= bit;
                b.white &= ~bit;
            },
        }
    }

    pub fn set(b: *@This(), x: anytype, y: anytype, c: Color) void {
        const bit: u64 = bitmask(x, y);
        setBitmask(b, bit, c);
    }
};

pub const init_board: Board = init: {
    var b: Board = .{ .white = 0, .black = 0 };
    b.set(3, 3, .white);
    b.set(4, 4, .white);
    b.set(3, 4, .black);
    b.set(4, 3, .black);
    break :init b;
};

pub fn computeValidSquares(b: Board, col: Color) u64 {
    assert(col != .empty);
    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;
    const empty = ~(own | opp);
    var valid: u64 = 0;

    // https://olinjohnson.github.io/posts/optimizing-chess-othello-and-connect-4-with-bitboards/

    const DIRECTIONS_SHIFTS = [8]i8{ -9, -8, -7, -1, 1, 7, 8, 9 };
    inline for (DIRECTIONS_SHIFTS) |dir| {
        const mask: u64 = switch (dir) {
            -9, -1, 7 => 0xFEFEFEFEFEFEFEFE,
            1, 9, -7 => 0x7F7F7F7F7F7F7F7F,
            -8, 8 => 0xFFFFFFFFFFFFFFFF,
            else => unreachable,
        };

        const udir: u6 = @intCast(@abs(dir));
        if (dir > 0) {
            var temp = opp & ((own & mask) << udir);
            inline for (0..5) |_| temp |= opp & ((temp & mask) << udir);
            valid |= empty & ((temp & mask) << udir);
        } else {
            var temp = opp & ((own & mask) >> udir);
            inline for (0..5) |_| temp |= opp & ((temp & mask) >> udir);
            valid |= empty & ((temp & mask) >> udir);
        }
    }

    return valid;
}

pub fn isValidPlay(b: Board, p: Coord, col: Color) bool {
    assert(col != .empty);
    const bit = bitmask(p[0], p[1]);
    return (computeValidSquares(b, col) & bit != 0);
}

pub fn playAt(b: Board, p: Coord, col: Color) Board {
    assert(col != .empty);
    const bit = bitmask(p[0], p[1]);
    if (std.debug.runtime_safety) assert(computeValidSquares(b, col) & bit != 0);

    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;

    var flipped: u64 = bit;

    const DIRECTIONS_SHIFTS = [8]i8{ -9, -8, -7, -1, 1, 7, 8, 9 };
    inline for (DIRECTIONS_SHIFTS) |dir| {
        const mask: u64 = switch (dir) {
            -9, -1, 7 => 0xFEFEFEFEFEFEFEFE,
            1, 9, -7 => 0x7F7F7F7F7F7F7F7F,
            -8, 8 => 0xFFFFFFFFFFFFFFFF,
            else => unreachable,
        };

        const udir: u6 = @intCast(@abs(dir));
        if (dir > 0) {
            var temp = opp & ((bit & mask) << udir);
            inline for (0..5) |_| temp |= opp & ((temp & mask) << udir);
            if (own & ((temp & mask) << udir) != 0) flipped |= temp;
        } else {
            var temp = opp & ((bit & mask) >> udir);
            inline for (0..5) |_| temp |= opp & ((temp & mask) >> udir);
            if (own & ((temp & mask) >> udir) != 0) flipped |= temp;
        }
    }

    if (col == .white) {
        return .{ .white = b.white | flipped, .black = b.black & ~flipped };
    } else {
        return .{ .white = b.white & ~flipped, .black = b.black | flipped };
    }
}

pub const Score = struct { whites: u32, blacks: u32 };
pub fn computeScore(b: Board) Score {
    return .{
        .whites = @popCount(b.white),
        .blacks = @popCount(b.black),
    };
}

// bon pas d'état persistant pour l'instant
pub const Engine = enum(i32) {
    none,
    random,
    greedy,
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
        .two_steps => computeStepBestMove(b, col, random, &ctx, 2).?.coord,
        .five_steps => computeStepBestMove(b, col, random, &ctx, 5).?.coord,
    };
}

fn computeRandomMove(b: Board, col: Color, random: std.Random) Coord {
    const valids = computeValidSquares(b, col);
    assert(valids != 0);

    while (true) {
        const index = random.intRangeAtMost(u6, 0, 63);
        const bit = @as(u64, 1) << index;
        if (valids & bit == 0) continue;
        return .{ index % 8, index / 8 };
    }
}

const EvalWeights = struct {
    piece_diff: i32 = 100,
    corner_occupancy: i32 = 1000,
    corner_closeness: i32 = 500,
    mobility: i32 = 50,
    stability: i32 = 200,
    endgame_piece_diff: i32 = 200,
};

const weights = EvalWeights{};

fn evaluation(b: Board, col: Color) i32 {
    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;

    const score = computeScore(b);
    const nb0: i32 = if (col == .white) @intCast(score.whites) else @intCast(score.blacks);
    const nb1: i32 = if (col == .black) @intCast(score.whites) else @intCast(score.blacks);
    const nbempty: i32 = @intCast(64 - (score.whites + score.blacks));

    var eval: i32 = (nb0 - nb1) * weights.piece_diff;

    // Occupation des coins
    const corner_occupancy: i32 = blk: {
        const CORNER_MASK: u64 = 0x8100000000000081; // Coins : a1, h1, a8, h8
        const own_corners: i32 = @intCast(@popCount(own & CORNER_MASK));
        const opp_corners: i32 = @intCast(@popCount(opp & CORNER_MASK));
        break :blk own_corners - opp_corners;
    };
    eval += corner_occupancy * weights.corner_occupancy;

    // Proximité des coins
    const corner_closeness: i32 = blk: {
        const X_SQUARE_MASK: u64 = 0x42C300000000C342; // Cases X adjacentes aux coins
        const own_x_squares: i32 = @intCast(@popCount(own & X_SQUARE_MASK));
        const opp_x_squares: i32 = @intCast(@popCount(opp & X_SQUARE_MASK));
        break :blk opp_x_squares - own_x_squares;
    };
    eval += corner_closeness * weights.corner_closeness;

    // Mobilité
    const mobility: i32 = blk: {
        const own_mobility: i32 = @intCast(@popCount(computeValidSquares(b, col)));
        const opp_mobility: i32 = @intCast(@popCount(computeValidSquares(b, col.next())));
        break :blk own_mobility - opp_mobility;
    };
    eval += mobility * weights.mobility;

    // Stabilité (bords)
    const stability: i32 = blk: {
        const EDGE_MASK: u64 = 0xFF818181818181FF; // Toutes les cases de bord
        const own_edges: i32 = @intCast(@popCount(own & EDGE_MASK));
        const opp_edges: i32 = @intCast(@popCount(opp & EDGE_MASK));
        break :blk own_edges - opp_edges;
    };
    eval += stability * weights.stability;

    // Considérations de fin de partie
    if (nbempty < 10) {
        eval += (nb0 - nb1) * weights.endgame_piece_diff;
    }

    return eval;
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
            const after = playAt(b, coord, col);
            const delta = evaluation(after, col);
            if (best == null or best_score < delta or (best_score == delta and random.boolean())) {
                best = coord;
                best_score = delta;
            }
        }
    }
    return if (best) |coord| .{ .coord = coord, .score = best_score } else null;
}

fn computeStepBestMove(b: Board, col: Color, random: ?std.Random, ctx: *Context, lookahead: u32) ?struct { coord: Coord, score: i32 } {
    const valids = computeValidSquares(b, col);
    if (valids == 0) {
        const is_main_move = random != null;
        if (is_main_move) return null;

        const valids2 = computeValidSquares(b, col.next());
        if (valids2 == 0) return null; // game over, l'adversaire pourra pas jouer non plus.
        if (computeStepBestMove(b, col.next(), null, ctx, lookahead)) |res| {
            return .{ .coord = undefined, .score = -res.score };
        } else return null;
    }
    var best: ?Coord = null;
    var best_score: i32 = undefined;
    for (0..64) |i| {
        const bit = @as(u64, 1) << @intCast(i);
        if (valids & bit == 0) continue;

        const coord: Coord = .{ @intCast(i % 8), @intCast(i / 8) };
        const after = playAt(b, coord, col);

        const expected: i32 = score: {
            if (ctx.dico.get(after)) |v| break :score v;
            if (lookahead > 0) {
                if (computeStepBestMove(after, col.next(), null, ctx, lookahead - 1)) |res|
                    break :score -res.score;
            }
            break :score evaluation(after, col);
        };
        ctx.dico.put(after, expected) catch unreachable;

        if (best != null and best_score == expected) {
            // choix aléatoire en cas d'égalité (juste pour l'appel initial, pas les exploration récursives)
            if (random) |rnd| {
                if (rnd.boolean())
                    best = null;
            }
        }
        if (best == null or best_score < expected) {
            best = coord;
            best_score = expected;
        }
    }
    return if (best) |coord| .{ .coord = coord, .score = best_score } else null;
}

// ================================================================
// Tests
// ================================================================

const dirsteps: [8 * 8][8][]const u64 = blk: {
    @setEvalBranchQuota(10000);
    // nb: testé avec u8 ibit pltuot que u64 mask.  -> 1.22x plus lent

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

fn playAtRef(b: Board, p: Coord, col: Color) !Board {
    assert(col != .empty);
    const bit0 = bitmask(p[0], p[1]);
    assert(computeValidSquares(b, col) & bit0 != 0);

    var b1 = b;
    b1.setBitmask(bit0, col);

    const occupied = b.white | b.black;
    const own = if (col == .white) &b1.white else &b1.black;
    const opp = if (col == .white) &b1.black else &b1.white;

    loop: for (&dirsteps[@intCast(p[1] * 8 + p[0])]) |steps| {
        for (steps) |bit| {
            if (occupied & bit == 0) continue :loop;
            if (own.* & bit != 0) break;
        } else continue :loop;

        for (steps) |bit| {
            if (opp.* & bit == 0) continue :loop;
            own.* |= bit;
            opp.* &= ~bit;
        }
    }
    return b1;
}

fn computeValidSquaresRef(b: Board, col: Color) u64 {
    assert(col != .empty);
    const occupied = b.white | b.black;
    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;
    var valid: u64 = 0;

    for (&dirsteps, 0..) |dirs, index| {
        const bit0: u64 = @as(u64, 1) << @intCast(index);
        if (occupied & bit0 != 0) continue;
        const ok: bool = loop: for (&dirs) |steps| {
            var has_oppo = false;
            for (steps) |bit| {
                if (own & bit != 0) {
                    if (has_oppo) break :loop true else continue :loop;
                }
                if (opp & bit != 0) {
                    has_oppo = true;
                } else {
                    continue :loop;
                }
            }
        } else false;

        if (ok) {
            valid |= bit0;
        }
    }
    return valid;
}

test "computeValidSquares" {
    const test_cases = &[_]Board{
        init_board,
        .{ .white = 0, .black = 0 },
        .{ .white = 1, .black = 0 },
        .{ .white = 0, .black = 1 },
        .{ .white = 0xFFFFFFFFFFFFFFFF, .black = 0x0 },
        .{ .white = 0x1111111111111110, .black = 0x1 },
        .{ .white = 0x0123456789ABCDEF, .black = 0x1240000000000010 },
        .{ .white = 0x7F7F70000F7F7F7E, .black = 0xFF0000001 },
        .{ .white = 0xFEFEFEFE00000EFE, .black = 0xFF00001 },
    };
    for (test_cases) |b| {
        const new_w = computeValidSquares(b, .white);
        const old_w = computeValidSquaresRef(b, .white);
        try std.testing.expectEqual(old_w, new_w);

        const new_b = computeValidSquares(b, .black);
        const old_b = computeValidSquaresRef(b, .black);
        try std.testing.expectEqual(old_b, new_b);
    }
}

test "computeValidSquares fuzz" {
    const input_bytes = std.testing.fuzzInput(.{});
    if (input_bytes.len != 16) return;
    var b: Board = std.mem.bytesAsValue(Board, input_bytes[0..16]).*;
    b.black &= ~b.white; // cleanu data

    const new_w = computeValidSquares(b, .white);
    const old_w = computeValidSquaresRef(b, .white);
    try std.testing.expectEqual(old_w, new_w);

    const new_b = computeValidSquares(b, .black);
    const old_b = computeValidSquaresRef(b, .black);
    try std.testing.expectEqual(old_b, new_b);
}

test "playAt fuzz" {
    const input_bytes = std.testing.fuzzInput(.{});
    if (input_bytes.len != 18) return;
    var b: Board = std.mem.bytesAsValue(Board, input_bytes[0..16]).*;
    b.black &= ~b.white; // cleanup data
    var p: Coord = std.mem.bytesAsValue(Coord, input_bytes[16..18]).*;
    p[0] &= 0x7;
    p[1] &= 0x7;

    const new_w = playAt(b, p, .white);
    const old_w = playAtRef(b, p, .white);
    try std.testing.expectEqual(old_w, new_w);

    const new_b = playAt(b, p, .black);
    const old_b = playAtRef(b, p, .black);
    try std.testing.expectEqual(old_b, new_b);
}
