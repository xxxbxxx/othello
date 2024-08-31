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

pub fn computeMovesEval(b: Board, col: Color, lookahead: u32) [64]i32 {
    const valids = computeValidSquares(b, col);
    if (valids == 0) {
        return [1]i32{0} ** 64;
    }

    const baseline = evaluate(b, col, .{});
    var evals = [1]i32{0} ** 64;
    for (0..64) |i| {
        const bit = @as(u64, 1) << @intCast(i);
        if (valids & bit == 0) continue;

        const coord: Coord = .{ @intCast(i % 8), @intCast(i / 8) };
        const after = playAt(b, coord, col);

        const new_eval = if (lookahead > 0) -computeRecursiveBestMove(after, col.next(), false, null, lookahead - 1, true, -INF, INF).eval else evaluate(after, col, .{});
        evals[i] = new_eval - baseline;
    }

    return evals;
}

// bon pas d'état persistant pour l'instant
pub const Engine = enum(i32) {
    none,
    random,
    greedy,
    small_lookahead,
    large_lookahead,
    mcts,
};

const INF: i32 = std.math.maxInt(i32);

pub fn computeBestMove(engine: Engine, b: Board, col: Color, alloc: std.mem.Allocator, random: std.Random) Coord {
    return switch (engine) {
        .none => unreachable,
        .random => computeRandomMove(b, col, random).?,
        .greedy => computeGreedyMove(b, col, random).coord.?,
        .small_lookahead => computeRecursiveBestMove(b, col, false, random, 2, true, -INF, INF).coord.?,
        .large_lookahead => computeRecursiveBestMove(b, col, false, random, 6, true, -INF, INF).coord.?,
        .mcts => MCTS.computeBestMove(b, col, 10_000, alloc, random),
    };
}

fn computeRandomMove(b: Board, col: Color, random: std.Random) ?Coord {
    const valids = computeValidSquares(b, col);
    if (valids == 0) return null;

    while (true) {
        const index = random.intRangeAtMost(u6, 0, 63);
        const bit = @as(u64, 1) << index;
        if (valids & bit == 0) continue;
        return .{ index % 8, index / 8 };
    }
}

const EvalWeights = struct {
    piece_diff: i32 = 10,
    corner_occupancy: i32 = 100,
    corner_closeness: i32 = 50,
    mobility: i32 = 5,
    stability: i32 = 20,
    endgame_piece_diff: i32 = 20,

    const finish: EvalWeights = .{
        .piece_diff = 1000,
        .corner_occupancy = 0,
        .corner_closeness = 0,
        .mobility = 0,
        .stability = 0,
        .endgame_piece_diff = 0,
    };
};

fn evaluate(b: Board, col: Color, weights: EvalWeights) i32 {
    const own = if (col == .white) b.white else b.black;
    const opp = if (col == .white) b.black else b.white;

    const own_count: i32 = @intCast(@popCount(own));
    const opp_count: i32 = @intCast(@popCount(opp));
    const empty_count: i32 = @intCast(64 - (own_count + opp_count));

    var eval: i32 = (own_count - opp_count) * weights.piece_diff;

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
    if (empty_count < 10) {
        eval += (own_count - opp_count) * weights.endgame_piece_diff;
    }

    return eval;
}

fn computeGreedyMove(b: Board, col: Color, random: ?std.Random) struct { coord: ?Coord, eval: i32 } {
    const valids = computeValidSquares(b, col);
    if (valids == 0)
        return .{ .coord = null, .eval = evaluate(b, col, .{}) };

    var best_coord: ?Coord = null;
    var best_eval: i32 = undefined;
    for (0..8) |y| {
        for (0..8) |x| {
            const bit = bitmask(x, y);
            if (valids & bit == 0) continue;

            const coord: Coord = .{ @intCast(x), @intCast(y) };
            const after = playAt(b, coord, col);
            const eval = evaluate(after, col, .{});

            if (best_coord != null and best_eval == eval) {
                if (random) |rnd| {
                    if (rnd.boolean())
                        best_coord = null;
                }
            }
            if (best_coord == null or best_eval < eval) {
                best_coord = coord;
                best_eval = eval;
            }
        }
    }
    return .{ .coord = best_coord.?, .eval = best_eval };
}

// ================================================================
// négamax avec élagage αβ
// ================================================================

fn computeRecursiveBestMove(b: Board, col: Color, skipped: bool, random: ?std.Random, lookahead: u32, comptime prune: bool, alpha: i32, beta: i32) struct { coord: ?Coord, eval: i32 } {
    const valids = computeValidSquares(b, col);
    if (valids == 0) {
        if (skipped) { // l'adversaire a aussiskippé son tour -> game over
            return .{ .coord = null, .eval = evaluate(b, col, EvalWeights.finish) };
        } else {
            const res = computeRecursiveBestMove(b, col.next(), true, null, lookahead, prune, -beta, -alpha);
            return .{ .coord = null, .eval = -res.eval };
        }
    }

    var alpha1 = alpha;
    var best_coord: ?Coord = null;
    var best_eval: i32 = undefined;
    for (0..64) |i| {
        const bit = @as(u64, 1) << @intCast(i);
        if (valids & bit == 0) continue;

        const coord: Coord = .{ @intCast(i % 8), @intCast(i / 8) };
        const after = playAt(b, coord, col);

        const eval: i32 = eval: {
            if (lookahead > 0) {
                const res = computeRecursiveBestMove(after, col.next(), false, null, lookahead - 1, prune, -beta, -alpha1);
                break :eval -res.eval;
            }
            break :eval evaluate(after, col, .{});
        };

        if (best_coord != null and best_eval == eval) {
            // choix aléatoire en cas d'égalité (juste pour l'appel initial, pas les explorations récursives afin que le pruning soit exact)
            if (random) |rnd| {
                if (rnd.boolean()) {
                    best_coord = null;
                }
            }
        }
        if (best_coord == null or best_eval < eval) {
            best_coord = coord;
            best_eval = eval;
        }
        alpha1 = @max(eval, alpha1);
        if (prune and alpha1 > beta) {
            // piège: alpha1 >= beta "marche" aussi, *mais* va renvoyer potentiellement une valeur égale à une évaluation précédente (alors que la vraie évaluation non-élaguée aurait été strictment pire)
            //        et comme en cas d'églité on va tirer au hasard, on va potentiellement choisir ce choix strictement moins bon mais qu'on a élagué dès l'égalité rencontrée...
            break;
        }
    }

    return .{ .coord = best_coord.?, .eval = best_eval };
}

// ================================================================
// monte carlo tree search
// ================================================================

const MCTS = struct {
    const Node = struct {
        move: union(enum) {
            play: Coord,
            skip: void,
            endgame: void,
        },
        wins: u32 = 0, // nb victoire du parent quand il choisit ce noeud et joue "move"
        visits: u32 = 0,
        children: []Node = &.{},
    };

    // Upper Confidence Bound
    fn uct(node: *Node, parent_visits: u32) f32 {
        const c = 1.41; // Exploration parameter  "theoretically equal to √2; in practice usually chosen empirically"
        if (node.visits == 0) return std.math.inf(f32);
        const w: f32 = @floatFromInt(node.wins);
        const n: f32 = @floatFromInt(node.visits);
        const N: f32 = @floatFromInt(parent_visits);
        return (w / n) + c * @sqrt(@log(N) / n);
    }

    fn getBestChild(node: *Node, visits: u32) *Node {
        var best_child: ?*Node = null;
        var best_uct: f32 = -std.math.inf(f32);
        for (node.children) |*child| {
            const child_uct = uct(child, visits);
            if (child_uct > best_uct) {
                best_uct = child_uct;
                best_child = child;
            }
        }
        return best_child.?;
    }

    // complète la feuille avec les coups possibles.
    fn expandLeaf(node: *Node, board: Board, color: Color, allocator: std.mem.Allocator) !void {
        assert(node.children.len == 0);
        const valids = computeValidSquares(board, color);

        if (valids == 0) {
            const nextvalids = computeValidSquares(board, color.next());
            if (nextvalids == 0) {
                const new_nodes: [1]Node = .{.{ .move = .endgame }};
                node.children = try allocator.dupe(Node, &new_nodes);
                return;
            }

            const new_nodes: [1]Node = .{.{ .move = .skip }};
            node.children = try allocator.dupe(Node, &new_nodes);
            return;
        }

        var new_nodes: [64]Node = undefined;
        var index: u32 = 0;
        for (0..64) |i| {
            const bit = @as(u64, 1) << @intCast(i);
            if (valids & bit == 0) continue;
            new_nodes[index] = .{ .move = .{ .play = .{ @intCast(i % 8), @intCast(i / 8) } } };
            index += 1;
        }
        assert(index > 0);
        node.children = try allocator.dupe(Node, new_nodes[0..index]);
    }

    // éhantillonne une fin de partie aléatoire et renvoie qui gagne
    fn sampleResult(board: Board, color: Color, random: std.Random) Color {
        var current_board = board;
        var current_color = color;
        var skipped = false;

        while (true) {
            if (computeRandomMove(current_board, current_color, random)) |move| {
                skipped = false;
                current_board = playAt(current_board, move, current_color);
            } else {
                if (skipped) break;
                skipped = true;
            }
            current_color = current_color.next();
        }

        const score = computeScore(current_board);
        if (score.whites < score.blacks) return .black;
        if (score.whites > score.blacks) return .white;
        return .empty;
    }

    fn recurse(node: *Node, board: Board, color: Color, allocator: std.mem.Allocator, random: std.Random) Color {
        if (node.visits == 0) {
            const winner = sampleResult(board, color, random);
            node.visits += 1;
            node.wins += @intFromBool(winner != color);
            return winner;
        }

        if (node.children.len == 0) {
            expandLeaf(node, board, color, allocator) catch @panic("OOM");
        }

        const best_child: *Node = getBestChild(node, node.visits);
        const winner = switch (best_child.move) {
            .endgame => sampleResult(board, color, random),
            .skip => recurse(best_child, board, color.next(), allocator, random),
            .play => |coord| res: {
                const new_board = playAt(board, coord, color);
                break :res recurse(best_child, new_board, color.next(), allocator, random);
            },
        };

        node.visits += 1;
        node.wins += @intFromBool(winner != color);
        return winner;
    }

    fn computeBestMove(board: Board, color: Color, iterations: u32, allocator: std.mem.Allocator, random: std.Random) Coord {
        var nodes_arena = std.heap.ArenaAllocator.init(allocator);
        defer nodes_arena.deinit();

        var root: Node = .{ .move = undefined };

        for (0..iterations) |_| {
            _ = recurse(&root, board, color, nodes_arena.allocator(), random);
        }

        // std.debug.print("mcts: {}/{}\n", .{ root.wins, root.visits });
        // for (root.children) |it| {
        //     std.debug.print("    @{any} -> {}/{}\n", .{ it.move.play, it.wins, it.visits });
        // }

        const best: *Node = getBestChild(&root, 1);
        return best.move.play;
    }
};

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

test "pruning" {
    const b = Board{ .white = 15699526974273773319, .black = 2449958358421798936 };

    const new_w = computeRecursiveBestMove(b, .white, false, null, 3, true, -INF, INF);
    const old_w = computeRecursiveBestMove(b, .white, false, null, 3, false, -INF, INF);
    try std.testing.expectEqual(old_w, new_w);
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

    if (computeValidSquares(b, .white) & bitmask(p[0], p[1]) != 0) {
        const new_w = playAt(b, p, .white);
        const old_w = playAtRef(b, p, .white);
        try std.testing.expectEqual(old_w, new_w);
    }

    if (computeValidSquares(b, .black) & bitmask(p[0], p[1]) != 0) {
        const new_b = playAt(b, p, .black);
        const old_b = playAtRef(b, p, .black);
        try std.testing.expectEqual(old_b, new_b);
    }
}

test "pruning fuzz" {
    const input_bytes = std.testing.fuzzInput(.{});
    if (input_bytes.len != 16) return;
    var b: Board = std.mem.bytesAsValue(Board, input_bytes[0..16]).*;
    b.black &= ~b.white; // cleanup data

    const depth = 3;

    const new_w = computeRecursiveBestMove(b, .white, false, null, depth, true, -INF, INF);
    const old_w = computeRecursiveBestMove(b, .white, false, null, depth, false, -INF, INF);
    try std.testing.expectEqual(old_w, new_w);

    const new_b = computeRecursiveBestMove(b, .black, false, null, depth, true, -INF, INF);
    const old_b = computeRecursiveBestMove(b, .black, false, null, depth, false, -INF, INF);
    try std.testing.expectEqual(old_b, new_b);
}

test "computeStepBestMove cas1" {
    const b: Board = .{ .black = 0x00002408180E0000, .white = 0x0000083000000000 };
    assert(b.black & b.white == 0);

    var prng = std.Random.DefaultPrng.init(std.testing.random_seed);
    const random = prng.random();

    const best = computeRecursiveBestMove(b, .white, false, random, 6, true, -INF, INF);
    try std.testing.expect(!std.mem.eql(i8, &best.coord.?, &.{ 6, 6 }));
}

test "computeStepBestMove cas2" {
    const b: Board = .{ .black = 0x000024040C1E2000, .white = 0x0000083810000000 };
    assert(b.black & b.white == 0);

    var prng = std.Random.DefaultPrng.init(std.testing.random_seed);
    const random = prng.random();

    const best = computeRecursiveBestMove(b, .white, false, random, 6, true, -INF, INF);
    try std.testing.expect(!std.mem.eql(i8, &best.coord.?, &.{ 6, 6 }));
}
