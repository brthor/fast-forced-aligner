// Research runtime fork of gpu_controls.cu; the recurrence arithmetic is unchanged.
// Native allocation/initialization, optional shared scratch and direct input addressing.
// Mode 0: conventional DP; 1: chunks without greedy; 2: full algorithm.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cmath>
#include <climits>
#include <type_traits>
#include <c10/cuda/CUDACachingAllocator.h>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

namespace {
template <typename scalar_t>
__device__ const scalar_t* emission_base(const scalar_t* packed, const int64_t* m, bool direct) {
  return direct ? reinterpret_cast<const scalar_t*>(m[3]) : packed + m[3];
}

// A unique framewise maximizer that matches exactly one expanded-transcript
// occurrence is a local optimum. Multiple occurrences or label ties use DP;
// no arbitrary occurrence is donated to seam reconciliation.
template <typename scalar_t, bool cooperative, bool compact_enabled>
__global__ void greedy_shortcuts(const scalar_t* emissions, const int* labels,
    const int64_t* metadata, int* greedy, int* relative, scalar_t* frame_scores,
    int* shortcuts, int64_t* stats, int* endpoints, scalar_t* objectives,
    int64_t* output, int H, int max_chunks, bool local_scratch, bool direct, int alphabet_size, bool compact_alphabet, const int* status) {
  const int record = blockIdx.y, chunk = blockIdx.x;
  if (status[record] >= 2) return;
  const auto* m = metadata + record * 8;
  const int T = m[0], S = m[1], V = m[2], blank = m[7];
  const int a = chunk * H, b = min(T, a + H), slot = record * max_chunks + chunk;
  if (a >= b) return;
  const auto* e = emission_base(emissions, m, direct);
  const auto* l = labels + m[4];
  extern __shared__ int alphabet[];
  // Each chunk owns its scratch. Use local indices rather than constructing
  // a pointer before the start of shared storage for nonzero chunk offsets.
  auto* g = local_scratch ? alphabet + alphabet_size : greedy + m[6];
  auto* r = local_scratch ? g + H : relative + m[6];
  auto* fs = local_scratch ? reinterpret_cast<scalar_t*>(r + H) : frame_scores + m[6];
  const int origin = local_scratch ? a : 0;
  __shared__ int bad, matches, start, alphabet_count;
  const bool compact = compact_enabled && V > 256;
  auto* symbols = alphabet + V;
  if (threadIdx.x == 0) { bad = 0; matches = 0; start = -1; alphabet_count = 0; }
  for (int c = threadIdx.x; c < V; c += blockDim.x) alphabet[c] = 0;
  __syncthreads();
  for (int s = threadIdx.x; s < S; s += blockDim.x) {
    if (atomicExch(alphabet + l[s], 1) == 0 && compact)
      symbols[atomicAdd(&alphabet_count, 1)] = l[s];
  }
  __syncthreads();
  if constexpr (cooperative) {
  // Pass 1: neighboring lanes read neighboring vocabulary entries. A warp
  // owns a frame, and the reduction tracks maximum multiplicity explicitly.
  // No score addition occurs here: finite unique maxima are bit-identical to
  // the serial vocabulary scan, while all ties still force ordinary DP.
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  for (int t = a + warp; t < b; t += warps) {
    scalar_t best = -INFINITY;
    int chosen = -1, count = 0;
    for (int index = lane; index < (compact ? alphabet_count : V); index += 32) {
      const int c = compact ? symbols[index] : index;
      if (!compact && !alphabet[c]) continue;
      const scalar_t score = e[static_cast<int64_t>(t) * V + c];
      if (score > best) { best = score; chosen = c; count = 1; }
      else if (score == best) { count = min(2, count + 1); }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      const scalar_t other = __shfl_down_sync(0xffffffff, best, offset);
      const int other_chosen = __shfl_down_sync(0xffffffff, chosen, offset);
      const int other_count = __shfl_down_sync(0xffffffff, count, offset);
      if (lane + offset < 32) {
        if (other > best) { best = other; chosen = other_chosen; count = other_count; }
        else if (other == best) count = min(2, count + other_count);
      }
    }
    if (lane == 0) {
      g[(t) - origin] = chosen;
      fs[(t) - origin] = best;
      if (!isfinite(best) || count != 1) atomicExch(&bad, 1);
    }
  }
  } else {
    // Pass 4 restores the frozen greedy scan, leaving pass 2's recurrence.
    for (int t = a + threadIdx.x; t < b; t += blockDim.x) {
      scalar_t best = -INFINITY;
      int chosen = -1;
      bool tie = false;
      for (int index = 0; index < (compact ? alphabet_count : V); ++index) {
        const int c = compact ? symbols[index] : index;
        if (!compact && !alphabet[c]) continue;
        const scalar_t score = e[static_cast<int64_t>(t) * V + c];
        if (score > best) { best = score; chosen = c; tie = false; }
        else if (score == best) tie = true;
      }
      g[(t) - origin] = chosen;
      fs[(t) - origin] = best;
      if (!isfinite(best) || tie) atomicExch(&bad, 1);
    }
  }
  __syncthreads();
  if (bad) return;
  if (threadIdx.x == 0) {
    r[(a) - origin] = 0;
    for (int t = a + 1; t < b; ++t)
      r[(t) - origin] = r[(t - 1) - origin] + (g[(t) - origin] == g[(t - 1) - origin] ? 0 :
          (g[(t) - origin] == blank || g[(t - 1) - origin] == blank ? 1 : 2));
  }
  __syncthreads();
  for (int s = threadIdx.x; s < S; s += blockDim.x) {
    if (l[s] != g[(a) - origin] || (a == 0 && s > 1)) continue;
    const int terminal = s + r[(b - 1) - origin];
    if (terminal >= S || (b == T && terminal < S - 2)) continue;
    bool legal = true;
    for (int t = a + 1; t < b; ++t) {
      if (g[(t) - origin] != g[(t - 1) - origin] && l[s + r[(t) - origin]] != g[(t) - origin]) { legal = false; break; }
    }
    if (legal) { atomicAdd(&matches, 1); atomicExch(&start, s); }
  }
  __syncthreads();
  if (matches != 1) {
    if (threadIdx.x == 0 && matches > 1)
      atomicAdd(reinterpret_cast<unsigned long long*>(stats + record * 3 + 2), 1ULL);
    return;
  }
  if (threadIdx.x == 0) {
    scalar_t score = fs[(a) - origin];
    for (int t = a + 1; t < b; ++t) score += fs[(t) - origin];
    // Overflow cannot be treated as a finite upper-bound certificate.
    if (!isfinite(score)) bad = 1;
    else {
      shortcuts[slot] = 1;
      objectives[slot] = score;
      endpoints[2 * slot] = start;
      endpoints[2 * slot + 1] = start + r[(b - 1) - origin];
      atomicAdd(reinterpret_cast<unsigned long long*>(stats + record * 3), 1ULL);
      atomicAdd(reinterpret_cast<unsigned long long*>(stats + record * 3 + 1),
                static_cast<unsigned long long>(b - a));
    }
  }
  __syncthreads();
  if (!bad) for (int t = a + threadIdx.x; t < b; t += blockDim.x)
    output[m[6] + t] = start + r[(t) - origin];
}

// Metadata: T, S, vocabulary, emission offset, state offset,
// backpointer offset, output offset, blank. Coordinates derive from block IDs.
__device__ bool legal_seam(int start, int before, const int* labels, int blank) {
  const int jump = start - before;
  return jump == 0 || jump == 1 ||
      (jump == 2 && labels[start] != blank && labels[start] != labels[start - 2]);
}

template <typename scalar_t>
__global__ void plan_repairs(const int* labels, const int64_t* metadata,
    const int* endpoints, const scalar_t* chunk_objectives, int* intervals,
    int* repair_passes, int64_t* repaired_frames, int max_chunks, int repair_budget_frames, int* early_done, bool overlap_fallback, int repair_min_frames, const int* status) {
  const int record = blockIdx.x;
  if (status[record] >= 2) return;
  const auto* m = metadata + record * 8;
  const auto* l = labels + m[4];
  if (overlap_fallback && early_done[3 * record] == 2) return;
  // Preflight before changing any intervals. A rejected round leaves original
  // endpoints and intervals intact so seam validation requests ordinary fallback.
  // Budget is the largest connected region's frame span, not a score heuristic.
  if (repair_budget_frames > 0 && m[0] >= repair_min_frames) {
    int before = -1, begin = 0, maximum_proposed = 0;
    bool reject = false;
    for (int c = 0; c < max_chunks; ++c) {
      const int k = record * max_chunks + c;
      intervals[3 * k + 2] = 0;
      if (intervals[3 * k] >= intervals[3 * k + 1]) continue;
      const bool join = before < 0 ||
          (isfinite(chunk_objectives[k]) && isfinite(chunk_objectives[before]) &&
           legal_seam(endpoints[2 * k], endpoints[2 * before + 1], l, m[7]));
      if (join) begin = intervals[3 * k];
      else {
        maximum_proposed = max(maximum_proposed, intervals[3 * k + 1] - begin);
        if (maximum_proposed > repair_budget_frames) reject = true;
      }
      before = k;
    }
    if (reject) {
      ++early_done[3 * record + 1];
      early_done[3 * record + 2] = max(early_done[3 * record + 2], maximum_proposed);
      if (overlap_fallback) early_done[3 * record] = 1;
      return;
    }
  }
  int previous = -1, group = -1;
  bool merged = false;
  for (int c = 0; c < max_chunks; ++c) {
    const int k = record * max_chunks + c;
    if (intervals[3 * k] >= intervals[3 * k + 1]) continue;
    intervals[3 * k + 2] = 0;
    const bool join = previous < 0 ||
        (isfinite(chunk_objectives[k]) && isfinite(chunk_objectives[previous]) &&
         legal_seam(endpoints[2 * k], endpoints[2 * previous + 1], l, m[7]));
    if (join) {
      group = k;
    } else {
      intervals[3 * group + 1] = intervals[3 * k + 1];
      intervals[3 * group + 2] = 1;
      intervals[3 * k + 1] = intervals[3 * k];
      merged = true;
    }
    // Endpoints remain those of the old pieces until the next kernel, even
    // when this slot was just absorbed into its connected merge component.
    previous = k;
  }
  if (merged) {
    ++repair_passes[record];
    for (int c = 0; c < max_chunks; ++c) {
      const int k = record * max_chunks + c;
      if (intervals[3 * k + 2])
        repaired_frames[record] += intervals[3 * k + 1] - intervals[3 * k];
    }
  }
}

// Pass 3 keeps striped DP scores in registers. Only two values per warp
// cross warp/stripe boundaries; ping-pong storage protects previous-row reads
// from the next-row writes before the single end-of-frame CTA barrier.
template <typename scalar_t, bool fallback, int capacity>
__device__ void register_recurrence(const scalar_t* e, const int* l,
    uint8_t* bp, int64_t* output, int* terminals, int* endpoints,
    scalar_t* chunk_objectives, int* status, scalar_t* objective,
    int record, int slot, int64_t output_offset,
    int T, int S, int V, int blank, int a, int b, scalar_t* tails, bool resume, int start_t, scalar_t* frontier, int* frontier_ends) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const int tail_count = 2 * warps * capacity;
  scalar_t* previous_tails = tails;
  scalar_t* next_tails = tails + tail_count;
  scalar_t values[capacity];
  int state_labels[capacity];
  bool can_skip[capacity];
  #pragma unroll
  for (int j = 0; j < capacity; ++j) {
    const int s = threadIdx.x + j * blockDim.x;
    state_labels[j] = s < S ? l[s] : blank;
    can_skip[j] = s >= 2 && s < S && state_labels[j] != blank && state_labels[j] != l[s - 2];
    values[j] = s < S ? (resume ? frontier[s] : ((a > 0 || s < 2)
        ? e[static_cast<int64_t>(a) * V + state_labels[j]] : -INFINITY)) : -INFINITY;
    if (s < S && !resume) bp[static_cast<int64_t>(a) * S + s] = 0;
    if (lane >= 30) previous_tails[2 * (j * warps + warp) + lane - 30] = values[j];
  }
  __syncthreads();
  for (int t = start_t; t < b; ++t) {
    #pragma unroll
    for (int j = 0; j < capacity; ++j) {
      const int s = threadIdx.x + j * blockDim.x;
      // All lanes, including padded states, participate in both shuffles.
      scalar_t step = __shfl_up_sync(0xffffffff, values[j], 1);
      scalar_t skip = __shfl_up_sync(0xffffffff, values[j], 2);
      const int preceding_warp = j * warps + warp - 1;
      if (lane == 0) step = preceding_warp >= 0 ? previous_tails[2 * preceding_warp + 1] : -INFINITY;
      if (lane < 2) skip = preceding_warp >= 0 ? previous_tails[2 * preceding_warp + lane] : -INFINITY;
      if (s < S) {
        scalar_t best = values[j];
        uint8_t jump = 0;
        if (s >= 1 && step > best) { best = step; jump = 1; }
        if (can_skip[j] && skip > best) { best = skip; jump = 2; }
        values[j] = best + e[static_cast<int64_t>(t) * V + state_labels[j]];
        bp[static_cast<int64_t>(t) * S + s] = jump;
      }
      if (lane >= 30) next_tails[2 * (j * warps + warp) + lane - 30] = values[j];
    }
    __syncthreads();
    auto* swap = previous_tails; previous_tails = next_tails; next_tails = swap;
  }
  if (frontier && !fallback) {
    #pragma unroll
    for (int j = 0; j < capacity; ++j) {
      const int s = threadIdx.x + j * blockDim.x;
      if (s < S) frontier[s] = values[j];
    }
    if (threadIdx.x == 0) frontier_ends[slot] = b;
  }
  __shared__ scalar_t terminal_scores[3]; // state zero, final token, final blank
  __shared__ scalar_t warp_scores[16];
  __shared__ int warp_states[16];
  scalar_t best = -INFINITY;
  int state = INT_MAX;
  #pragma unroll
  for (int j = 0; j < capacity; ++j) {
    const int s = threadIdx.x + j * blockDim.x;
    if (s == 0) terminal_scores[0] = values[j];
    if (s == S - 2) terminal_scores[1] = values[j];
    if (s == S - 1) terminal_scores[2] = values[j];
    if (s < S && (values[j] > best || (values[j] == best && s < state))) {
      best = values[j]; state = s;
    }
  }
  if (b != T) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      const scalar_t other = __shfl_down_sync(0xffffffff, best, offset);
      const int other_state = __shfl_down_sync(0xffffffff, state, offset);
      if (lane + offset < 32 && (other > best || (other == best && other_state < state))) {
        best = other; state = other_state;
      }
    }
    if (lane == 0) { warp_scores[warp] = best; warp_states[warp] = state; }
  }
  __syncthreads();
  if (b != T && threadIdx.x < 32) {
    best = lane < warps ? warp_scores[lane] : -INFINITY;
    state = lane < warps ? warp_states[lane] : INT_MAX;
    for (int offset = 16; offset > 0; offset >>= 1) {
      const scalar_t other = __shfl_down_sync(0xffffffff, best, offset);
      const int other_state = __shfl_down_sync(0xffffffff, state, offset);
      if (lane + offset < 32 && (other > best || (other == best && other_state < state))) {
        best = other; state = other_state;
      }
    }
  }
  if (threadIdx.x == 0) {
    int terminal = state;
    scalar_t score = best;
    if (b == T) {
      terminal = S - 1; score = terminal_scores[2];
      if (S > 1 && terminal_scores[1] >= terminal_scores[2]) {
        terminal = S - 2; score = terminal_scores[1];
      }
    } else if (isnan(terminal_scores[0])) {
      terminal = 0; score = terminal_scores[0];
    }
    if (fallback) {
      terminals[record] = terminal;
      objective[record] = score;
      status[record] = isfinite(score) ? 0 : 1;
    } else {
      chunk_objectives[slot] = score;
      endpoints[2 * slot + 1] = terminal;
      int s = terminal;
      for (int t = b - 1; t >= a; --t) {
        output[output_offset + t] = s;
        s -= bp[static_cast<int64_t>(t) * S + s];
      }
      endpoints[2 * slot] = s;
    }
  }
}

template <typename scalar_t, bool fallback, int capacity = 0, bool registers = false>
__device__ void recurrence_impl(const scalar_t* emissions, const int* labels,
    const int64_t* metadata, uint8_t* pointers, int* terminals,
    int* endpoints, scalar_t* chunk_objectives, int64_t* output,
    int* status, scalar_t* objective, const int* stitched, const int* intervals,
    bool repairs, int H, int max_chunks, const int* shortcuts, bool initial, bool direct, int record_override, bool frontier_reuse, scalar_t* frontiers, int* frontier_ends, int64_t* reuse_stats, int frontier_min_frames) {
  const int record = record_override >= 0 ? record_override : (fallback ? blockIdx.x : blockIdx.y);
  if (status[record] >= 2 || (fallback && stitched[record])) return;
  const auto* m = metadata + record * 8;
  const int T = m[0], S = m[1], V = m[2], blank = m[7];
  const int chunk = fallback ? 0 : blockIdx.x;
  const int slot = record * max_chunks + chunk;
  if (!fallback && initial && shortcuts[slot]) return;
  const int a = fallback ? 0 : (repairs ? intervals[3 * slot] : chunk * H);
  const int b = fallback ? T : (repairs ? intervals[3 * slot + 1] : min(T, a + H));
  if (a >= b || (!fallback && repairs && !intervals[3 * slot + 2])) return;
  // An interval's surviving slot retains its original start. Merges only
  // extend its end; absorbed slots never execute again. Full fallback uses
  // slot zero, whose original start is zero. A zero end denotes no DP frontier
  // (including every greedy-only interval). Prefix backpointers stay intact.
  frontier_reuse = frontier_reuse && T >= frontier_min_frames;
  const int saved_end = frontier_reuse ? frontier_ends[slot] : 0;
  const bool resume = saved_end > a && saved_end <= b;
  const int start_t = resume ? saved_end : a + 1;
  scalar_t* frontier = frontier_reuse ? frontiers + m[4] * max_chunks + static_cast<int64_t>(chunk) * S : nullptr;
  if (frontier_reuse && threadIdx.x == 0) {
    if (resume) {
      atomicAdd(reinterpret_cast<unsigned long long*>(reuse_stats + 4 * record + (fallback ? 1 : 0)), static_cast<unsigned long long>(saved_end - a));
      atomicAdd(reinterpret_cast<unsigned long long*>(reuse_stats + 4 * record + 3), static_cast<unsigned long long>(S));
    }
    if (!fallback) atomicAdd(reinterpret_cast<unsigned long long*>(reuse_stats + 4 * record + 2), static_cast<unsigned long long>(S));
  }
  const auto* e = emission_base(emissions, m, direct);
  const auto* l = labels + m[4];
  auto* bp = pointers + m[5];
  extern __shared__ double shared_storage[];
  if constexpr (registers) {
    register_recurrence<scalar_t, fallback, capacity>(e, l, bp, output, terminals,
        endpoints, chunk_objectives, status, objective, record, slot, m[6],
        T, S, V, blank, a, b, reinterpret_cast<scalar_t*>(shared_storage), resume, start_t, frontier, frontier_reuse ? frontier_ends : nullptr);
    return;
  }
  auto* prev = reinterpret_cast<scalar_t*>(shared_storage);
  auto* next = prev + S;
  // Pass 2 caches invariant transcript data in bounded striped register tiles.
  // Capacity zero preserves the pass-1 implementation and launch behavior.
  if constexpr (capacity > 0) {
    int state_labels[capacity];
    bool can_skip[capacity];
    #pragma unroll
    for (int j = 0; j < capacity; ++j) {
      const int s = threadIdx.x + j * blockDim.x;
      state_labels[j] = s < S ? l[s] : blank;
      can_skip[j] = s >= 2 && s < S && state_labels[j] != blank && state_labels[j] != l[s - 2];
      if (s < S) {
        prev[s] = resume ? frontier[s] : ((a > 0 || s < 2) ? e[static_cast<int64_t>(a) * V + state_labels[j]] : -INFINITY);
        if (!resume) bp[static_cast<int64_t>(a) * S + s] = 0;
      }
    }
    __syncthreads();
    for (int t = start_t; t < b; ++t) {
      #pragma unroll
      for (int j = 0; j < capacity; ++j) {
        const int s = threadIdx.x + j * blockDim.x;
        if (s < S) {
          scalar_t best = prev[s];
          uint8_t jump = 0;
          if (s >= 1 && prev[s - 1] > best) { best = prev[s - 1]; jump = 1; }
          if (can_skip[j] && prev[s - 2] > best) { best = prev[s - 2]; jump = 2; }
          next[s] = best + e[static_cast<int64_t>(t) * V + state_labels[j]];
          bp[static_cast<int64_t>(t) * S + s] = jump;
        }
      }
      __syncthreads();
      auto* swap = prev; prev = next; next = swap;
    }
  } else {
  for (int s = threadIdx.x; s < S; s += blockDim.x) {
    prev[s] = resume ? frontier[s] : ((a > 0 || s < 2) ? e[static_cast<int64_t>(a) * V + l[s]] : -INFINITY);
    if (!resume) bp[static_cast<int64_t>(a) * S + s] = 0;
  }
  __syncthreads();
  for (int t = start_t; t < b; ++t) {
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
      scalar_t best = prev[s];
      uint8_t jump = 0;
      if (s >= 1 && prev[s - 1] > best) { best = prev[s - 1]; jump = 1; }
      if (s >= 2 && l[s] != blank && l[s] != l[s - 2] && prev[s - 2] > best) {
        best = prev[s - 2]; jump = 2;
      }
      next[s] = best + e[static_cast<int64_t>(t) * V + l[s]];
      bp[static_cast<int64_t>(t) * S + s] = jump;
    }
    __syncthreads();
    auto* swap = prev; prev = next; next = swap;
  }
  }
  if (frontier_reuse && !fallback) {
    for (int s = threadIdx.x; s < S; s += blockDim.x) frontier[s] = prev[s];
    if (threadIdx.x == 0) frontier_ends[slot] = b;
  }
  int reduced_terminal = 0;
  if constexpr (capacity > 0) {
    // A stable max/argmax does not reassociate score arithmetic. Padding and
    // NaNs cannot displace a real state; the original serial scan's special
    // behavior when state zero is NaN is restored by thread zero below.
    __shared__ scalar_t warp_scores[16];
    __shared__ int warp_states[16];
    if (b != T) {
      scalar_t best = -INFINITY;
      int state = INT_MAX;
      #pragma unroll
      for (int j = 0; j < capacity; ++j) {
        const int s = threadIdx.x + j * blockDim.x;
        if (s < S && (prev[s] > best || (prev[s] == best && s < state))) {
          best = prev[s]; state = s;
        }
      }
      const int lane = threadIdx.x & 31;
      for (int offset = 16; offset > 0; offset >>= 1) {
        const scalar_t other = __shfl_down_sync(0xffffffff, best, offset);
        const int other_state = __shfl_down_sync(0xffffffff, state, offset);
        if (lane + offset < 32 && (other > best || (other == best && other_state < state))) {
          best = other; state = other_state;
        }
      }
      if (lane == 0) { warp_scores[threadIdx.x >> 5] = best; warp_states[threadIdx.x >> 5] = state; }
      __syncthreads();
      if (threadIdx.x < 32) {
        best = lane < (blockDim.x >> 5) ? warp_scores[lane] : -INFINITY;
        state = lane < (blockDim.x >> 5) ? warp_states[lane] : INT_MAX;
        for (int offset = 16; offset > 0; offset >>= 1) {
          const scalar_t other = __shfl_down_sync(0xffffffff, best, offset);
          const int other_state = __shfl_down_sync(0xffffffff, state, offset);
          if (lane + offset < 32 && (other > best || (other == best && other_state < state))) {
            best = other; state = other_state;
          }
        }
        reduced_terminal = isnan(prev[0]) ? 0 : state;
      }
    }
  }
  if (threadIdx.x == 0) {
    int terminal = S - 1;
    if (b == T) {
      if (S > 1 && prev[S - 2] >= prev[S - 1]) terminal = S - 2;
    } else {
      if constexpr (capacity > 0) terminal = reduced_terminal;
      else {
        terminal = 0;
        for (int s = 1; s < S; ++s) if (prev[s] > prev[terminal]) terminal = s;
      }
    }
    if (fallback) {
      terminals[record] = terminal;
      objective[record] = prev[terminal];
      status[record] = isfinite(prev[terminal]) ? 0 : 1;
    } else {
      const int k = record * max_chunks + chunk;
      chunk_objectives[k] = prev[terminal];
      endpoints[2 * k + 1] = terminal;
      int s = terminal;
      for (int t = b - 1; t >= a; --t) {
        output[m[6] + t] = s;
        s -= bp[static_cast<int64_t>(t) * S + s];
      }
      endpoints[2 * k] = s;
    }
  }
}

template <typename scalar_t, bool fallback, int capacity = 0, bool registers = false>
__global__ void recurrence(const scalar_t* emissions, const int* labels,
    const int64_t* metadata, uint8_t* pointers, int* terminals,
    int* endpoints, scalar_t* chunk_objectives, int64_t* output,
    int* status, scalar_t* objective, const int* stitched, const int* intervals,
    bool repairs, int H, int max_chunks, const int* shortcuts, bool initial, bool direct, bool frontier_reuse, scalar_t* frontiers, int* frontier_ends, int64_t* reuse_stats, int frontier_min_frames) {
  recurrence_impl<scalar_t, fallback, capacity, registers>(emissions, labels, metadata, pointers, terminals, endpoints, chunk_objectives, output, status, objective, stitched, intervals, repairs, H, max_chunks, shortcuts, initial, direct, -1, frontier_reuse, frontiers, frontier_ends, reuse_stats, frontier_min_frames);
}

template <typename scalar_t, bool fallback, int capacity = 0, bool registers = false>
__global__ void repair_recurrence(const scalar_t* emissions, const int* labels,
    const int64_t* metadata, uint8_t* pointers, int* terminals,
    int* endpoints, scalar_t* chunk_objectives, int64_t* output,
    int* status, scalar_t* objective, const int* stitched, const int* intervals,
    bool repairs, int H, int max_chunks, const int* shortcuts, bool initial, bool direct, int* early_done, bool overlap_fallback, bool frontier_reuse, scalar_t* frontiers, int* frontier_ends, int64_t* reuse_stats, int frontier_min_frames) {
  const int record = blockIdx.y;
  if (overlap_fallback && early_done[3 * record]) {
    if (blockIdx.x != 0 || early_done[3 * record] == 2) return;
    recurrence_impl<scalar_t, true, capacity, registers>(emissions, labels, metadata, pointers, terminals, endpoints, chunk_objectives, output, status, objective, stitched, intervals, repairs, H, max_chunks, shortcuts, initial, direct, record, frontier_reuse, frontiers, frontier_ends, reuse_stats, frontier_min_frames);
    __syncthreads();
    if (threadIdx.x == 0) early_done[3 * record] = 2;
    return;
  }
  recurrence_impl<scalar_t, false, capacity, registers>(emissions, labels, metadata, pointers, terminals, endpoints, chunk_objectives, output, status, objective, stitched, intervals, repairs, H, max_chunks, shortcuts, initial, direct, -1, frontier_reuse, frontiers, frontier_ends, reuse_stats, frontier_min_frames);
}

template <typename scalar_t, bool fallback, int capacity = 0, bool registers = false>
__global__ void fallback_recurrence(const scalar_t* emissions, const int* labels,
    const int64_t* metadata, uint8_t* pointers, int* terminals,
    int* endpoints, scalar_t* chunk_objectives, int64_t* output,
    int* status, scalar_t* objective, const int* stitched, const int* intervals,
    bool repairs, int H, int max_chunks, const int* shortcuts, bool initial, bool direct, const int* early_done, bool overlap_fallback, bool frontier_reuse, scalar_t* frontiers, int* frontier_ends, int64_t* reuse_stats, int frontier_min_frames) {
  if (overlap_fallback && early_done[3 * blockIdx.x] == 2) return;
  recurrence_impl<scalar_t, true, capacity, registers>(emissions, labels, metadata, pointers, terminals, endpoints, chunk_objectives, output, status, objective, stitched, intervals, repairs, H, max_chunks, shortcuts, initial, direct, -1, frontier_reuse, frontiers, frontier_ends, reuse_stats, frontier_min_frames);
}

template <typename scalar_t>
__global__ void validate_seams(const int* labels, const int64_t* metadata,
    const int* endpoints, const scalar_t* chunk_objectives, int* stitched,
    int* status, scalar_t* objective, const int* intervals, int* repair_passes,
    int64_t* repaired_frames, bool repairs, int H, int max_chunks, const int* early_done, bool overlap_fallback) {
  const int record = blockIdx.x;
  if (status[record] >= 2) return;
  const auto* m = metadata + record * 8;
  const int T = m[0], n = 1 + (T - 1) / H;
  // A completed early fallback owns terminals/status/objective. Its stitched
  // value remains zero, so final traceback uses its conventional backpointers.
  if (overlap_fallback && early_done[3 * record] == 2) return;
  const auto* l = labels + m[4];
  bool legal = true;
  scalar_t total = 0;
  int previous = -1;
  if (!repairs) { repair_passes[record] = 0; repaired_frames[record] = 0; }
  for (int c = 0; c < n; ++c) {
    const int k = record * max_chunks + c;
    if (repairs && intervals[3 * k] >= intervals[3 * k + 1]) continue;
    const scalar_t cost = chunk_objectives[k];
    legal = legal && isfinite(cost);
    total += cost;
    if (previous >= 0)
      legal = legal && legal_seam(endpoints[2 * k], endpoints[2 * previous + 1], l, m[7]);
    previous = k;
  }
  legal = legal && isfinite(total);
  stitched[record] = legal ? 1 : 0;
  status[record] = 0;
  objective[record] = total;
}

// Traceback parallelism is shared engineering: conventional and speculative
// fallback use exactly the same option. Every ancestor lookup is derived from
// the current call's already-computed backpointers inside measured execution.
constexpr int TRACE_H = 128;
__global__ void trace_ancestors(const int64_t* metadata, const uint8_t* pointers,
    const int* status, const int* stitched, int16_t* ancestors, int trace_chunks, int trace_min_frames) {
  const int record = blockIdx.y, chunk = blockIdx.x;
  const auto* m = metadata + 8 * record;
  const int T = m[0], S = m[1];
  const int a = chunk * TRACE_H, b = static_cast<int>(min(static_cast<int64_t>(T), static_cast<int64_t>(a) + TRACE_H));
  if (stitched[record] || status[record] || T < trace_min_frames || a >= b) return;
  const auto* bp = pointers + m[5];
  auto* out = ancestors + m[4] * trace_chunks + static_cast<int64_t>(chunk) * S;
  for (int terminal = threadIdx.x; terminal < S; terminal += blockDim.x) {
    int state = terminal;
    for (int t = b - 1; t >= a; --t) state -= bp[static_cast<int64_t>(t) * S + state];
    out[terminal] = static_cast<int16_t>(state);
  }
}
__global__ void trace_anchors(const int64_t* metadata, const int* terminals,
    const int* status, const int* stitched, const int16_t* ancestors,
    int* anchors, int trace_chunks, int trace_min_frames) {
  const int record = blockIdx.x;
  const auto* m = metadata + 8 * record;
  const int T = m[0], S = m[1];
  if (stitched[record] || status[record] || T < trace_min_frames) return;
  int state = terminals[record];
  for (int chunk = (T - 1) / TRACE_H; chunk >= 0; --chunk) {
    anchors[record * trace_chunks + chunk] = state;
    state = ancestors[m[4] * trace_chunks + static_cast<int64_t>(chunk) * S + state];
  }
}
__global__ void trace_pieces(const int64_t* metadata, const uint8_t* pointers,
    const int* status, const int* stitched, const int* anchors,
    int64_t* output, int trace_chunks, int trace_min_frames) {
  const int record = blockIdx.y, chunk = blockIdx.x;
  const auto* m = metadata + 8 * record;
  const int T = m[0], S = m[1], a = chunk * TRACE_H, b = static_cast<int>(min(static_cast<int64_t>(T), static_cast<int64_t>(a) + TRACE_H));
  if (stitched[record] || status[record] || T < trace_min_frames || a >= b) return;
  const auto* bp = pointers + m[5];
  int state = anchors[record * trace_chunks + chunk];
  for (int t = b - 1; t >= a; --t) {
    output[m[6] + t] = state;
    state -= bp[static_cast<int64_t>(t) * S + state];
  }
}

template <typename scalar_t>
__global__ void finish(const scalar_t* emissions, const int* labels,
    const int64_t* metadata, const uint8_t* pointers, const int* terminals,
    const int* status, const int* stitched, int64_t* output, scalar_t* scores, bool direct, bool parallel_traceback, int trace_min_frames) {
  const int record = blockIdx.x;
  const auto* m = metadata + record * 8;
  const int T = m[0], S = m[1], V = m[2];
  auto* path = output + m[6];
  auto* sc = scores + m[6];
  if (status[record] != 0) {
    for (int t = threadIdx.x; t < T; t += blockDim.x) { path[t] = -1; sc[t] = -INFINITY; }
    return;
  }
  if (!stitched[record] && !(parallel_traceback && T >= trace_min_frames) && threadIdx.x == 0) {
    int s = terminals[record];
    for (int t = T - 1; t >= 0; --t) {
      path[t] = s;
      s -= pointers[m[5] + static_cast<int64_t>(t) * S + s];
    }
  }
  __syncthreads();
  for (int t = threadIdx.x; t < T; t += blockDim.x) {
    const int label = labels[m[4] + path[t]];
    path[t] = label;
    sc[t] = emission_base(emissions, m, direct)[static_cast<int64_t>(t) * V + label];
  }
}

void launch_alignment(const at::Tensor& emissions, const at::Tensor& labels,
    const at::Tensor& metadata, const at::Tensor& pointers, const at::Tensor& terminals,
    const at::Tensor& endpoints, const at::Tensor& chunk_objectives, const at::Tensor& intervals,
    const at::Tensor& output, const at::Tensor& scores, const at::Tensor& status,
    const at::Tensor& objective, const at::Tensor& stitched, const at::Tensor& repair_passes,
    const at::Tensor& repaired_frames, const at::Tensor& greedy_labels,
    const at::Tensor& relative_states, const at::Tensor& greedy_scores,
    const at::Tensor& shortcuts, const at::Tensor& shortcut_stats,
    int64_t max_vocabulary, int64_t max_states, int64_t threads, int64_t H,
    int64_t max_chunks, int64_t repair_rounds, int64_t optimization_pass, int64_t mode,
    bool local_scratch, bool direct, int repair_budget_frames, int repair_threads, bool compact_alphabet, const at::Tensor& trace_storage, const at::Tensor& trace_anchor_storage,
    int trace_chunks, bool parallel_traceback, const at::Tensor& early_done, bool overlap_fallback, int trace_min_frames, int repair_min_frames, bool frontier_reuse, const at::Tensor& frontiers, const at::Tensor& frontier_ends, const at::Tensor& reuse_stats, int frontier_min_frames) {
  TORCH_CHECK(mode >= 0 && mode <= 2, "invalid research control mode");
  TORCH_CHECK((optimization_pass >= 1 && optimization_pass <= 4), "optimization pass must be 1, 2, 3, or 4");
  TORCH_CHECK(emissions.is_cuda() && emissions.is_contiguous(), "CUDA contiguous emissions required");
  TORCH_CHECK(max_states >= 1 && max_states <= 4096, "state width outside supported range");
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512, "invalid block size");
  TORCH_CHECK(H > 0 && max_chunks > 0, "invalid chunk shape");
  TORCH_CHECK(repair_rounds >= 0 && repair_rounds <= 2, "invalid repair rounds");
  c10::cuda::CUDAGuard guard(emissions.device());
  const int batch = metadata.size(0);
  const auto stream = at::cuda::getCurrentCUDAStream();
  {
    using scalar_t = float;
    const int alphabet_size = max_vocabulary * (compact_alphabet ? 2 : 1);
    auto launch_greedy = [&](auto cooperative_tag, auto compact_tag) {
    greedy_shortcuts<scalar_t, decltype(cooperative_tag)::value, decltype(compact_tag)::value><<<dim3(max_chunks, batch), threads,
        (alphabet_size + (local_scratch ? 3 * H : 0)) * sizeof(int), stream>>>(emissions.data_ptr<scalar_t>(),
        labels.data_ptr<int>(), metadata.data_ptr<int64_t>(), greedy_labels.data_ptr<int>(),
        relative_states.data_ptr<int>(), greedy_scores.data_ptr<scalar_t>(),
        shortcuts.data_ptr<int>(), shortcut_stats.data_ptr<int64_t>(), endpoints.data_ptr<int>(),
        chunk_objectives.data_ptr<scalar_t>(), output.data_ptr<int64_t>(), H, max_chunks, local_scratch, direct, alphabet_size, compact_alphabet, status.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if (mode == 2) {
      if (compact_alphabet) {
        if (optimization_pass == 4) launch_greedy(std::false_type{}, std::true_type{});
        else launch_greedy(std::true_type{}, std::true_type{});
      } else {
        if (optimization_pass == 4) launch_greedy(std::false_type{}, std::false_type{});
        else launch_greedy(std::true_type{}, std::false_type{});
      }
    }
    auto launch_recurrences = [&](auto capacity_tag, auto register_tag) {
    constexpr int capacity = decltype(capacity_tag)::value;
    constexpr bool registers = decltype(register_tag)::value;
    const int shared_bytes = (registers ? 4 * capacity * (threads / 32) : 2 * max_states) * sizeof(scalar_t);
    if (shared_bytes > 49152) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(recurrence<scalar_t, false, capacity, registers>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
      C10_CUDA_CHECK(cudaFuncSetAttribute(recurrence<scalar_t, true, capacity, registers>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    }
    if (mode != 0) {
    recurrence<scalar_t, false, capacity, registers><<<dim3(max_chunks, batch), threads, shared_bytes, stream>>>(
        emissions.data_ptr<scalar_t>(), labels.data_ptr<int>(), metadata.data_ptr<int64_t>(),
        pointers.data_ptr<uint8_t>(), terminals.data_ptr<int>(), endpoints.data_ptr<int>(),
        chunk_objectives.data_ptr<scalar_t>(), output.data_ptr<int64_t>(), status.data_ptr<int>(),
        objective.data_ptr<scalar_t>(), stitched.data_ptr<int>(), intervals.data_ptr<int>(), repair_rounds > 0, H, max_chunks, shortcuts.data_ptr<int>(), true, direct, frontier_reuse, frontiers.data_ptr<scalar_t>(), frontier_ends.data_ptr<int>(), reuse_stats.data_ptr<int64_t>(), frontier_min_frames);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    for (int round = 0; round < repair_rounds; ++round) {
      plan_repairs<scalar_t><<<batch, 1, 0, stream>>>(labels.data_ptr<int>(), metadata.data_ptr<int64_t>(),
          endpoints.data_ptr<int>(), chunk_objectives.data_ptr<scalar_t>(), intervals.data_ptr<int>(),
          repair_passes.data_ptr<int>(), repaired_frames.data_ptr<int64_t>(), max_chunks, repair_budget_frames, early_done.data_ptr<int>(), overlap_fallback, repair_min_frames, status.data_ptr<int>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      // Repair geometry is independent of initial/fallback geometry. Dispatch
      // its own stripe capacity: reusing the initial capacity could omit states.
      const int rt = repair_threads ? repair_threads : threads;
      auto launch_repair = [&](auto repair_capacity_tag) {
        constexpr int rc = decltype(repair_capacity_tag)::value;
        const int repair_shared = (registers ? 4 * rc * (rt / 32) : 2 * max_states) * sizeof(scalar_t);
        if (overlap_fallback) {
        repair_recurrence<scalar_t, false, rc, registers><<<dim3(max_chunks, batch), rt, repair_shared, stream>>>(
            emissions.data_ptr<scalar_t>(), labels.data_ptr<int>(), metadata.data_ptr<int64_t>(),
            pointers.data_ptr<uint8_t>(), terminals.data_ptr<int>(), endpoints.data_ptr<int>(),
            chunk_objectives.data_ptr<scalar_t>(), output.data_ptr<int64_t>(), status.data_ptr<int>(),
            objective.data_ptr<scalar_t>(), stitched.data_ptr<int>(), intervals.data_ptr<int>(), true, H, max_chunks, shortcuts.data_ptr<int>(), false, direct, early_done.data_ptr<int>(), overlap_fallback, frontier_reuse, frontiers.data_ptr<scalar_t>(), frontier_ends.data_ptr<int>(), reuse_stats.data_ptr<int64_t>(), frontier_min_frames);
        } else {
        recurrence<scalar_t, false, rc, registers><<<dim3(max_chunks, batch), rt, repair_shared, stream>>>(
            emissions.data_ptr<scalar_t>(), labels.data_ptr<int>(), metadata.data_ptr<int64_t>(),
            pointers.data_ptr<uint8_t>(), terminals.data_ptr<int>(), endpoints.data_ptr<int>(),
            chunk_objectives.data_ptr<scalar_t>(), output.data_ptr<int64_t>(), status.data_ptr<int>(),
            objective.data_ptr<scalar_t>(), stitched.data_ptr<int>(), intervals.data_ptr<int>(), true, H, max_chunks, shortcuts.data_ptr<int>(), false, direct, frontier_reuse, frontiers.data_ptr<scalar_t>(), frontier_ends.data_ptr<int>(), reuse_stats.data_ptr<int64_t>(), frontier_min_frames);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      };
      const int repair_slots = (max_states + rt - 1) / rt;
      if (repair_slots <= 1) launch_repair(std::integral_constant<int, 1>{});
      else if (repair_slots <= 2) launch_repair(std::integral_constant<int, 2>{});
      else if (repair_slots <= 4) launch_repair(std::integral_constant<int, 4>{});
      else if (repair_slots <= 8) launch_repair(std::integral_constant<int, 8>{});
      else if (repair_slots <= 16) launch_repair(std::integral_constant<int, 16>{});
      else launch_repair(std::integral_constant<int, 32>{});
    }
    validate_seams<scalar_t><<<batch, 1, 0, stream>>>(labels.data_ptr<int>(), metadata.data_ptr<int64_t>(),
        endpoints.data_ptr<int>(), chunk_objectives.data_ptr<scalar_t>(), stitched.data_ptr<int>(),
        status.data_ptr<int>(), objective.data_ptr<scalar_t>(), intervals.data_ptr<int>(),
        repair_passes.data_ptr<int>(), repaired_frames.data_ptr<int64_t>(), repair_rounds > 0, H, max_chunks, early_done.data_ptr<int>(), overlap_fallback);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    } // Chunk and repair stages are absent from conventional DP.
    if (overlap_fallback) {
    fallback_recurrence<scalar_t, true, capacity, registers><<<batch, threads, shared_bytes, stream>>>(
        emissions.data_ptr<scalar_t>(), labels.data_ptr<int>(), metadata.data_ptr<int64_t>(),
        pointers.data_ptr<uint8_t>(), terminals.data_ptr<int>(), endpoints.data_ptr<int>(),
        chunk_objectives.data_ptr<scalar_t>(), output.data_ptr<int64_t>(), status.data_ptr<int>(),
        objective.data_ptr<scalar_t>(), stitched.data_ptr<int>(), intervals.data_ptr<int>(), false, H, max_chunks, shortcuts.data_ptr<int>(), false, direct, early_done.data_ptr<int>(), overlap_fallback, frontier_reuse, frontiers.data_ptr<scalar_t>(), frontier_ends.data_ptr<int>(), reuse_stats.data_ptr<int64_t>(), frontier_min_frames);
    } else {
    recurrence<scalar_t, true, capacity, registers><<<batch, threads, shared_bytes, stream>>>(
        emissions.data_ptr<scalar_t>(), labels.data_ptr<int>(), metadata.data_ptr<int64_t>(),
        pointers.data_ptr<uint8_t>(), terminals.data_ptr<int>(), endpoints.data_ptr<int>(),
        chunk_objectives.data_ptr<scalar_t>(), output.data_ptr<int64_t>(), status.data_ptr<int>(),
        objective.data_ptr<scalar_t>(), stitched.data_ptr<int>(), intervals.data_ptr<int>(), false, H, max_chunks, shortcuts.data_ptr<int>(), false, direct, frontier_reuse, frontiers.data_ptr<scalar_t>(), frontier_ends.data_ptr<int>(), reuse_stats.data_ptr<int64_t>(), frontier_min_frames);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    auto dispatch_capacity = [&](auto register_tag) {
      const int slots = optimization_pass == 1 ? 0 : (max_states + threads - 1) / threads;
      if constexpr (!decltype(register_tag)::value) {
        if (slots == 0) { launch_recurrences(std::integral_constant<int, 0>{}, register_tag); return; }
      }
      if (slots <= 1) launch_recurrences(std::integral_constant<int, 1>{}, register_tag);
      else if (slots <= 2) launch_recurrences(std::integral_constant<int, 2>{}, register_tag);
      else if (slots <= 4) launch_recurrences(std::integral_constant<int, 4>{}, register_tag);
      else if (slots <= 8) launch_recurrences(std::integral_constant<int, 8>{}, register_tag);
      else if (slots <= 16) launch_recurrences(std::integral_constant<int, 16>{}, register_tag);
      else launch_recurrences(std::integral_constant<int, 32>{}, register_tag);
    };
    if (optimization_pass == 3) dispatch_capacity(std::true_type{});
    else dispatch_capacity(std::false_type{});
    if (parallel_traceback) {
      trace_ancestors<<<dim3(trace_chunks, batch), 256, 0, stream>>>(metadata.data_ptr<int64_t>(),
          pointers.data_ptr<uint8_t>(), status.data_ptr<int>(), stitched.data_ptr<int>(),
          trace_storage.data_ptr<int16_t>(), trace_chunks, trace_min_frames);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      trace_anchors<<<batch, 1, 0, stream>>>(metadata.data_ptr<int64_t>(), terminals.data_ptr<int>(),
          status.data_ptr<int>(), stitched.data_ptr<int>(), trace_storage.data_ptr<int16_t>(),
          trace_anchor_storage.data_ptr<int>(), trace_chunks, trace_min_frames);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      trace_pieces<<<dim3(trace_chunks, batch), 1, 0, stream>>>(metadata.data_ptr<int64_t>(),
          pointers.data_ptr<uint8_t>(), status.data_ptr<int>(), stitched.data_ptr<int>(),
          trace_anchor_storage.data_ptr<int>(), output.data_ptr<int64_t>(), trace_chunks, trace_min_frames);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    finish<scalar_t><<<batch, threads, 0, stream>>>(emissions.data_ptr<scalar_t>(), labels.data_ptr<int>(),
        metadata.data_ptr<int64_t>(), pointers.data_ptr<uint8_t>(), terminals.data_ptr<int>(),
        status.data_ptr<int>(), stitched.data_ptr<int>(), output.data_ptr<int64_t>(), scores.data_ptr<scalar_t>(), direct, parallel_traceback, trace_min_frames);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}


template <bool check_inputs>
__global__ void pack_inputs(const int64_t* descriptors, float* emissions,
                            int* expanded, int batch, int* intervals, int* stitched, int* repair_passes,
                            int64_t* repaired_frames, int* shortcuts, int64_t* stats,
                            int H, int max_chunks, int mode, bool direct, int* early_done, bool frontier_reuse, int* frontier_ends, int64_t* reuse_stats, const bool* emission_validity, int* status, float* objective) {
  const int record = blockIdx.y;
  const auto* m = descriptors + 8 * record;
  const auto* sources = descriptors + 8 * batch + 2 * record;
  const auto* source_emissions = reinterpret_cast<const float*>(sources[0]);
  const auto* source_targets = reinterpret_cast<const int64_t*>(sources[1]);
  const int64_t elements = m[0] * m[2];
  // A rectangular grid handles ragged recordings without a host launch per item.
  if (!direct) for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < elements; i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    emissions[m[3] + i] = source_emissions[i];
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      if (frontier_reuse) for (int j = 0; j < 4; ++j) reuse_stats[4 * record + j] = 0;
      status[record] = emission_validity && !emission_validity[record] ? 2 : 0;
      objective[record] = -INFINITY;
      stitched[record] = 0;
      early_done[3 * record] = early_done[3 * record + 1] = early_done[3 * record + 2] = 0;
      repair_passes[record] = 0;
      repaired_frames[record] = 0;
      stats[3 * record] = stats[3 * record + 1] = stats[3 * record + 2] = 0;
    }
    if (mode != 0) for (int chunk = threadIdx.x; chunk < max_chunks; chunk += blockDim.x) {
      const int k = record * max_chunks + chunk;
      intervals[3 * k] = min(static_cast<int>(m[0]), chunk * H);
      intervals[3 * k + 1] = min(static_cast<int>(m[0]), (chunk + 1) * H);
      intervals[3 * k + 2] = 1;
      shortcuts[k] = 0;
      if (frontier_reuse) frontier_ends[k] = 0;
    }
    // Encode invalid-ID counts above the repeat count, which is at most 2046.
    // Validate int64 IDs before narrowing: e.g. 2**32 + 1 must not become 1.
    int validation = 0;
    for (int s = threadIdx.x; s < m[1]; s += blockDim.x) {
      const int64_t token = (s & 1) ? source_targets[s / 2] : m[7];
      expanded[m[4] + s] = static_cast<int>(token);
      if constexpr (check_inputs) {
        if (s & 1) {
          validation += (token < 0 || token >= m[2] || token == m[7]) ? 4096 : 0;
          validation += s > 1 && token == source_targets[s / 2 - 1];
        }
      }
    }
    if constexpr (check_inputs) {
      using Reduction = cub::BlockReduce<int, 256>;
      __shared__ typename Reduction::TempStorage storage;
      const int combined = Reduction(storage).Sum(validation);
      if (threadIdx.x == 0) {
        // Target errors take precedence over the optional emission mask.
        // Subsequent kernels skip statuses >= 2 before any label-indexed read.
        if (combined >= 4096) status[record] = 3;
        else if (m[1] / 2 + combined > m[0]) status[record] = 4;
      }
    }
  }
}

// One CTA per record avoids atomics and host completion checks. Each CTA scans
// its entire recording; no caller-visible metadata is read back from the GPU.
__global__ void emission_validity_kernel(const int64_t* descriptors, bool* valid) {
  const auto* d = descriptors + 2 * blockIdx.x;
  const auto* values = reinterpret_cast<const float*>(d[0]);
  bool invalid = false;
  for (int64_t i = threadIdx.x; i < d[1]; i += blockDim.x) {
    const float value = values[i];
    invalid = invalid || isnan(value) || value == INFINITY;
  }
  const int any_invalid = __syncthreads_or(invalid);
  if (threadIdx.x == 0) valid[blockIdx.x] = !any_invalid;
}

at::Tensor emission_validity(at::TensorList emissions) {
  const int64_t count = emissions.size();
  TORCH_CHECK(count > 0 && count <= 65535, "provide 1..65535 emission tensors");
  TORCH_CHECK(emissions[0].is_cuda(), "CUDA emissions required");
  const auto device = emissions[0].device();
  c10::cuda::CUDAGuard guard(device);
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto host = at::empty({count, 2}, at::TensorOptions().dtype(at::kLong).device(at::kCPU).pinned_memory(true));
  auto* descriptors = host.data_ptr<int64_t>();
  for (int64_t i = 0; i < count; ++i) {
    const auto& e = emissions[i];
    TORCH_CHECK(e.device() == device && e.scalar_type() == at::kFloat &&
                e.dim() == 2 && e.is_contiguous() && e.size(0) > 0 &&
                e.size(0) <= INT_MAX && e.size(1) > 0 && e.size(1) <= 8192,
                "emissions must be contiguous CUDA FP32 [T,V] on one device with supported dimensions");
    descriptors[2 * i] = reinterpret_cast<int64_t>(e.data_ptr<float>());
    descriptors[2 * i + 1] = e.numel();
    c10::cuda::CUDACachingAllocator::recordStream(e.storage().data_ptr(), stream);
  }
  auto resident = host.to(device, /*non_blocking=*/true);
  auto result = at::empty({count}, emissions[0].options().dtype(at::kBool));
  emission_validity_kernel<<<count, 256, 0, stream>>>(resident.data_ptr<int64_t>(), result.data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}

std::tuple<std::vector<at::Tensor>, std::vector<int64_t>> align(
    at::TensorList emissions, at::TensorList targets, at::IntArrayRef blanks,
    int64_t chunk_frames, int64_t mode, int64_t threads, int64_t repair_rounds,
    int64_t optimization_pass, int64_t variant, int64_t repair_budget_frames,
    int64_t repair_threads, bool compact_alphabet, bool parallel_traceback, bool overlap_fallback, int64_t trace_min_frames, int64_t repair_min_frames, bool frontier_reuse, int64_t frontier_min_frames, const std::optional<at::Tensor>& emission_validity, bool check_inputs) {
  TORCH_CHECK(frontier_min_frames >= 0 && frontier_min_frames <= INT_MAX, "frontier_min_frames must be a nonnegative int32");
  TORCH_CHECK(repair_min_frames >= 0 && repair_min_frames <= INT_MAX, "repair_min_frames must be a nonnegative int32");
  TORCH_CHECK(trace_min_frames > 0 && trace_min_frames <= INT_MAX, "trace_min_frames must be a positive int32");
  TORCH_CHECK(!overlap_fallback || repair_threads == 0 || repair_threads == threads, "overlap preserves fallback geometry");
  TORCH_CHECK(repair_budget_frames >= 0 && repair_budget_frames <= INT_MAX, "repair budget must fit nonnegative int32");
  TORCH_CHECK(repair_threads == 0 || repair_threads == 128 || repair_threads == 256 || repair_threads == 512,
              "repair threads must be 0/128/256/512");
  TORCH_CHECK(variant >= 0 && variant <= 2, "variant must be 0, 1, or 2");
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512, "invalid block size");
  TORCH_CHECK(optimization_pass == 2 || optimization_pass == 3, "optimization pass must be 2 or 3");
  TORCH_CHECK(repair_rounds >= 0 && repair_rounds <= 2, "invalid repair rounds");
  frontier_reuse = frontier_reuse && mode != 0;
  const bool direct = variant == 2;
  const int64_t count = emissions.size();
  TORCH_CHECK(count > 0 && count <= 65535 && targets.size() == count && blanks.size() == count,
              "batch must contain 1..65535 matching emissions, targets and blanks");
  TORCH_CHECK(chunk_frames > 0 && chunk_frames <= std::numeric_limits<int>::max() &&
              mode >= 0 && mode <= 2, "invalid chunk size or mode");
  TORCH_CHECK(emissions[0].is_cuda(), "CUDA emissions required");
  const auto device = emissions[0].device();
  c10::cuda::CUDAGuard guard(device);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (emission_validity.has_value()) {
    const auto& mask = *emission_validity;
    TORCH_CHECK(mask.device() == device && mask.scalar_type() == at::kBool &&
                mask.dim() == 1 && mask.numel() == count && mask.is_contiguous(),
                "emission_validity must be contiguous CUDA bool [batch] on the input device");
    c10::cuda::CUDACachingAllocator::recordStream(mask.storage().data_ptr(), stream);
  }
  // Pinned staging permits one asynchronous metadata/pointer upload. This allocation,
  // host population, and transfer all belong to the measured preparation operation.
  auto host = at::empty({count * 10}, at::TensorOptions().dtype(at::kLong).device(at::kCPU).pinned_memory(true));
  auto* records = host.data_ptr<int64_t>();
  std::vector<int64_t> sizes;
  sizes.reserve(count);
  int64_t total_emissions = 0, total_states = 0, total_cells = 0, total_frames = 0;
  int64_t max_states = 0, max_vocab = 0, max_chunks = 0, max_elements = 0, max_frames = 0;
  for (int64_t i = 0; i < count; ++i) {
    const auto& e = emissions[i];
    const auto& t = targets[i];
    TORCH_CHECK(e.device() == device && t.device() == device &&
                e.scalar_type() == at::kFloat && t.scalar_type() == at::kLong &&
                e.dim() == 2 && t.dim() == 1 && e.is_contiguous() && t.is_contiguous(),
                "resident inputs must be contiguous FP32 [T,V] and int64 [U] on one CUDA device");
    const int64_t frames = e.size(0), vocab = e.size(1), states = 2 * t.numel() + 1;
    TORCH_CHECK(frames > 0 && frames <= std::numeric_limits<int>::max() &&
                states <= 4096 && vocab > 0 && vocab <= 8192 &&
                blanks[i] >= 0 && blanks[i] < vocab, "resident input shape or blank outside supported range");
    auto* m = records + 8 * i;
    m[0] = frames; m[1] = states; m[2] = vocab; m[3] = direct ? reinterpret_cast<int64_t>(e.data_ptr<float>()) : total_emissions;
    m[4] = total_states; m[5] = total_cells; m[6] = total_frames; m[7] = blanks[i];
    records[8 * count + 2 * i] = reinterpret_cast<int64_t>(e.data_ptr<float>());
    records[8 * count + 2 * i + 1] = reinterpret_cast<int64_t>(t.data_ptr<int64_t>());
    // The kernel consumes raw addresses, so protect source allocations even if
    // their tensors are released on another stream before all consumers finish.
    c10::cuda::CUDACachingAllocator::recordStream(e.storage().data_ptr(), stream);
    if (t.numel()) c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(), stream);
    total_emissions += e.numel(); total_states += states;
    total_cells += frames * states; total_frames += frames;
    max_states = std::max(max_states, states); max_vocab = std::max(max_vocab, vocab);
    max_chunks = std::max(max_chunks, 1 + (frames - 1) / chunk_frames);
    max_elements = std::max(max_elements, e.numel());
    max_frames = std::max(max_frames, frames);
    sizes.push_back(frames);
  }
  // Budget/overlap setup must amortize over sufficiently long records. Shape
  // inspection is already inside this call. Entirely shorter batches retain the
  // original-shaped repair/fallback kernels; mixed batches gate each record in
  // plan_repairs while sharing the selected launch geometry.
  if (max_frames < repair_min_frames) {
    repair_budget_frames = 0;
    overlap_fallback = false;
  }
  // The shared recurrence uses int chunk endpoints and three-column interval
  // indices. Reject overflow before allocating any packed GPU work buffers.
  TORCH_CHECK(max_chunks <= std::numeric_limits<int>::max() / chunk_frames &&
              max_chunks <= std::numeric_limits<int>::max() / (3 * count),
              "chunk endpoints or interval indices exceed int32 range");
  frontier_reuse = frontier_reuse && max_frames >= frontier_min_frames;
  const auto options = emissions[0].options();
  auto empty = [&](at::IntArrayRef shape, at::ScalarType type) { return at::empty(shape, options.dtype(type)); };
  auto descriptors = host.to(device, /*non_blocking=*/true);
  auto packed_emissions = empty({direct ? 0 : total_emissions}, at::kFloat);
  auto expanded = empty({total_states}, at::kInt);
  // Keep shared scratch bounded at 32 KiB including the alphabet. Larger
  // chunk/vocabulary combinations use the exact original global scratch path.
  // Two alphabet arrays are only needed for the optional compact scan.
  // The maximum vocabulary can exceed the default shared-memory budget with
  // two arrays; then use the unmodified masked scan rather than changing bounds.
  compact_alphabet = compact_alphabet && max_vocab > 256 && max_vocab <= 4096;
  const int alphabet_slots = max_vocab * (compact_alphabet ? 2 : 1);
  const bool local_scratch = variant >= 1 && alphabet_slots + 3 * chunk_frames <= 8192;
  const int64_t chunks = mode != 0 ? count : 0;
  const int64_t greedy_frames = mode == 2 && !local_scratch ? total_frames : 0;
  auto metadata = descriptors.narrow(0, 0, count * 8).view({count, 8});
  auto pointers = empty({total_cells}, at::kByte);
  auto terminals = empty({count}, at::kInt);
  auto endpoints = empty({chunks, max_chunks, 2}, at::kInt);
  auto chunk_objectives = empty({chunks, max_chunks}, at::kFloat);
  auto intervals = empty({chunks, max_chunks, 3}, at::kInt);
  auto greedy_labels = empty({greedy_frames}, at::kInt);
  auto relative_states = empty({greedy_frames}, at::kInt);
  auto greedy_scores = empty({greedy_frames}, at::kFloat);
  auto shortcuts = empty({chunks, max_chunks}, at::kInt);
  auto output = empty({total_frames}, at::kLong);
  auto scores = empty({total_frames}, at::kFloat);
  auto status = empty({count}, at::kInt);
  auto objective = empty({count}, at::kFloat);
  auto stitched = empty({count}, at::kInt);
  auto early_done = empty({count, 3}, at::kInt);
  auto frontiers = empty({frontier_reuse ? total_states * max_chunks : 0}, at::kFloat);
  auto frontier_ends = empty({frontier_reuse ? count * max_chunks : 0}, at::kInt);
  auto reuse_stats = empty({frontier_reuse ? count : 0, 4}, at::kLong);
  auto repair_passes = empty({count}, at::kInt);
  auto repaired_frames = empty({count}, at::kLong);
  auto shortcut_stats = empty({count, 3}, at::kLong);
  parallel_traceback = parallel_traceback && max_frames >= trace_min_frames;
  const int64_t trace_chunks = (max_frames - 1) / TRACE_H + 1;
  auto trace_storage = parallel_traceback ? empty({total_states * trace_chunks}, at::kShort) : at::Tensor();
  auto trace_anchor_storage = parallel_traceback ? empty({count * trace_chunks}, at::kInt) : at::Tensor();
  const unsigned blocks = direct ? 1 : static_cast<unsigned>(std::min<int64_t>(32, (max_elements + 255) / 256));
  const auto pack = check_inputs ? pack_inputs<true> : pack_inputs<false>;
  pack<<<dim3(blocks, count), 256, 0, stream>>>(
      descriptors.data_ptr<int64_t>(), packed_emissions.data_ptr<float>(), expanded.data_ptr<int>(), count,
      intervals.data_ptr<int>(), stitched.data_ptr<int>(), repair_passes.data_ptr<int>(),
      repaired_frames.data_ptr<int64_t>(), shortcuts.data_ptr<int>(), shortcut_stats.data_ptr<int64_t>(),
      chunk_frames, max_chunks, mode, direct, early_done.data_ptr<int>(), frontier_reuse, frontier_ends.data_ptr<int>(), reuse_stats.data_ptr<int64_t>(),
      emission_validity.has_value() ? emission_validity->data_ptr<bool>() : nullptr, status.data_ptr<int>(), objective.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  launch_alignment(packed_emissions, expanded, metadata, pointers, terminals,
      endpoints, chunk_objectives, intervals, output, scores, status, objective,
      stitched, repair_passes, repaired_frames, greedy_labels, relative_states,
      greedy_scores, shortcuts, shortcut_stats, max_vocab, max_states, threads,
      chunk_frames, max_chunks, repair_rounds, optimization_pass, mode, local_scratch, direct, repair_budget_frames, repair_threads, compact_alphabet,
      trace_storage, trace_anchor_storage, trace_chunks, parallel_traceback, early_done, overlap_fallback, trace_min_frames, repair_min_frames, frontier_reuse, frontiers, frontier_ends, reuse_stats, frontier_min_frames);
  return {{output, scores, status, objective, stitched, repair_passes, repaired_frames, shortcut_stats, early_done, reuse_stats},
          std::move(sizes)};
}
} // namespace

TORCH_LIBRARY(fast_forced_aligner_cuda, m) {
  m.def("emission_validity(Tensor[] emissions) -> Tensor");
  m.def("align(Tensor[] emissions, Tensor[] targets, int[] blanks, int chunk_frames, int mode, "
        "int threads, int repair_rounds, int optimization_pass, int variant, "
        "int repair_budget_frames, int repair_threads, bool compact_alphabet, bool parallel_traceback, bool overlap_fallback, int trace_min_frames, int repair_min_frames, bool frontier_reuse, int frontier_min_frames, Tensor? emission_validity=None, bool check_inputs=True) -> (Tensor[], int[])");
}
TORCH_LIBRARY_IMPL(fast_forced_aligner_cuda, CUDA, m) { m.impl("align", &align); m.impl("emission_validity", &emission_validity); }
