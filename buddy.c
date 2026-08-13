#include "buddy.h"

/*
 * Buddy allocator implementation.
 *
 * The pool is described by a complete binary tree of blocks.
 *   - root (id 0)        : rank = maxrank, covers the whole pool
 *   - node id            : left child = 2*id+1, right child = 2*id+2
 *   - rank of a node     : maxrank - level, where level is the depth (root=0)
 *   - a block of rank r  : covers 2^(r-1) pages
 *
 * A block is in one of four states:
 *   UNUSED : not currently an active node (its parent is the active block)
 *   FREE   : an active free block (leaf in the active partition; may be big)
 *   ALLOC  : an active allocated leaf
 *   SPLIT  : an active internal node whose two children are active
 *
 * Free blocks are kept on per-rank doubly linked free lists so insertion and
 * removal are O(1).
 */

#define MAXRANK_H 16                 /* hard cap from the problem statement */
#define MAX_T    ((1 << MAXRANK_H) - 1)   /* 65535, max number of blocks   */
#define MAX_N    (1 << (MAXRANK_H - 1))   /* 32768, max pool pages         */

#define ST_UNUSED 0
#define ST_FREE   1
#define ST_ALLOC  2
#define ST_SPLIT  3

static unsigned char *g_base = 0;
static int            g_npages = 0;
static int            g_maxrank = 0;     /* pool max rank */

/* per-block metadata, indexed by node id (0 .. 2^maxrank-2) */
static char  g_state[MAX_T + 1];
static int   g_page[MAX_T + 1];          /* page offset of the block start  */
static char  g_rank[MAX_T + 1];          /* rank of the block               */

/* free lists: one doubly linked list per rank */
static int   g_free_head[MAXRANK_H + 1];
static int   g_free_next[MAX_T + 1];
static int   g_free_prev[MAX_T + 1];
static int   g_free_cnt[MAXRANK_H + 1];  /* count of free blocks per rank    */

/* for an allocated leaf, the node id that starts at this page (else -1) */
static int   g_owner[MAX_N];

static void add_free(int id) {
    int r = g_rank[id];
    g_free_cnt[r]++;
    g_free_next[id] = g_free_head[r];
    g_free_prev[id] = -1;
    if (g_free_head[r] != -1) g_free_prev[g_free_head[r]] = id;
    g_free_head[r] = id;
}

static void remove_free(int id) {
    int r = g_rank[id];
    g_free_cnt[r]--;
    if (g_free_prev[id] == -1) g_free_head[r] = g_free_next[id];
    else                       g_free_next[g_free_prev[id]] = g_free_next[id];
    if (g_free_next[id] != -1) g_free_prev[g_free_next[id]] = g_free_prev[id];
    g_free_next[id] = -1;
    g_free_prev[id] = -1;
}

int init_page(void *p, int pgcount) {
    g_base = (unsigned char *)p;
    g_npages = pgcount;

    /* pool max rank: pgcount must be a power of two; rank = log2(pgcount)+1 */
    int k = 0, n = pgcount;
    while (n > 1) { n >>= 1; k++; }
    g_maxrank = k + 1;

    int total = (1 << g_maxrank) - 1;   /* number of nodes in the tree */

    for (int i = 0; i <= MAX_T; i++) {
        g_state[i] = ST_UNUSED;
        g_free_next[i] = -1;
        g_free_prev[i] = -1;
    }
    for (int r = 0; r <= MAXRANK_H; r++) {
        g_free_head[r] = -1;
        g_free_cnt[r] = 0;
    }
    for (int i = 0; i < MAX_N; i++) g_owner[i] = -1;

    /* build the tree: rank and page offset of every node */
    g_rank[0] = (char)g_maxrank;
    g_page[0] = 0;
    for (int id = 0; id < total; id++) {
        int r = g_rank[id];
        if (r > 1) {                     /* internal node -> has children */
            int child_rank = r - 1;
            int half_pages = 1 << (child_rank - 1);
            int l = 2 * id + 1;
            int rr = 2 * id + 2;
            g_rank[l] = (char)child_rank;  g_page[l] = g_page[id];
            g_rank[rr] = (char)child_rank; g_page[rr] = g_page[id] + half_pages;
        }
    }

    /* whole pool starts as one free block */
    g_state[0] = ST_FREE;
    add_free(0);

    return OK;
}

void *alloc_pages(int rank) {
    if (rank < 1 || rank > g_maxrank) return (void *)(-EINVAL);

    /* find the smallest free rank that can satisfy the request */
    int k = rank;
    while (k <= g_maxrank && g_free_head[k] == -1) k++;
    if (k > g_maxrank) return (void *)(-ENOSPC);

    int id = g_free_head[k];
    remove_free(id);

    /* split down, always keeping the lower (left) half for the allocation */
    while (k > rank) {
        int rchild = 2 * id + 2;
        int lchild = 2 * id + 1;
        g_state[id] = ST_SPLIT;
        g_state[rchild] = ST_FREE;
        add_free(rchild);
        id = lchild;
        k--;
    }

    g_state[id] = ST_ALLOC;
    g_owner[g_page[id]] = id;
    return g_base + (unsigned long)g_page[id] * 4096UL;
}

int return_pages(void *p) {
    if (p == 0) return -EINVAL;
    unsigned long off = (unsigned char *)p - g_base;
    if (off % 4096 != 0) return -EINVAL;
    if (off / 4096 >= (unsigned long)g_npages) return -EINVAL;
    int pg = (int)(off / 4096);

    int id = g_owner[pg];
    if (id < 0) return -EINVAL;          /* not the start of an allocated block */
    g_owner[pg] = -1;

    g_state[id] = ST_FREE;
    add_free(id);

    /* coalesce with buddy while the buddy is also free */
    while (id != 0) {
        int pid = (id - 1) / 2;
        int bid = (id % 2 == 1) ? (id + 1) : (id - 1);
        if (g_state[bid] == ST_FREE) {
            remove_free(bid);
            remove_free(id);
            g_state[bid] = ST_UNUSED;
            id = pid;
            g_state[id] = ST_FREE;
            add_free(id);
        } else {
            break;
        }
    }
    return OK;
}

int query_ranks(void *p) {
    if (p == 0) return -EINVAL;
    unsigned long off = (unsigned char *)p - g_base;
    if (off % 4096 != 0) return -EINVAL;
    if (off / 4096 >= (unsigned long)g_npages) return -EINVAL;
    int pg = (int)(off / 4096);

    int id = 0;
    while (g_state[id] == ST_SPLIT) {
        int r = g_rank[id];
        int half = 1 << (r - 2);          /* pages covered by the left child */
        int start = g_page[id];
        if (pg < start + half) id = 2 * id + 1;
        else                   id = 2 * id + 2;
    }
    return g_rank[id];
}

int query_page_counts(int rank) {
    if (rank < 1 || rank > g_maxrank) return -EINVAL;
    return g_free_cnt[rank];
}
