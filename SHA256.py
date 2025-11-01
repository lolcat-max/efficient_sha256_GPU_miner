import pyopencl as cl
import numpy as np
import time

# Optimized parallel OpenCL kernel for SHA-256 mining
opencl_sha256_kernel = """
// SHA-256 Constants
__constant uint K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

void sha256_transform(uchar *input, int len, uchar *output) {
    uint h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    uint w[64];
    uint a, b, c, d, e, f, g, h_temp;
    uint t1, t2;
    
    uchar block[64];
    for (int i = 0; i < 64; i++) block[i] = 0;
    for (int i = 0; i < len && i < 64; i++) block[i] = input[i];
    
    if (len < 56) {
        block[len] = 0x80;
        block[63] = (len * 8) & 0xff;
        block[62] = ((len * 8) >> 8) & 0xff;
    }
    
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint)block[i*4] << 24) | ((uint)block[i*4+1] << 16) | 
               ((uint)block[i*4+2] << 8) | ((uint)block[i*4+3]);
    }
    for (int i = 16; i < 64; i++) {
        w[i] = SIG1(w[i-2]) + w[i-7] + SIG0(w[i-15]) + w[i-16];
    }
    
    a = h[0]; b = h[1]; c = h[2]; d = h[3];
    e = h[4]; f = h[5]; g = h[6]; h_temp = h[7];
    
    for (int i = 0; i < 64; i++) {
        t1 = h_temp + EP1(e) + CH(e, f, g) + K[i] + w[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h_temp = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += h_temp;
    
    for (int i = 0; i < 8; i++) {
        output[i*4]   = (h[i] >> 24) & 0xff;
        output[i*4+1] = (h[i] >> 16) & 0xff;
        output[i*4+2] = (h[i] >> 8) & 0xff;
        output[i*4+3] = h[i] & 0xff;
    }
}

__kernel void mine_sha256(__global uint *found_nonce, __global uchar *found_hash,
                          __global int *found_flag, __global uint *stats_buffer,
                          uint start_nonce, uint target_zeros) {
    int gid = get_global_id(0);
    uint nonce = start_nonce + gid;
    
    // Convert nonce to string
    uchar nonce_str[16];
    int len = 0;
    uint temp = nonce;
    if (temp == 0) {
        nonce_str[len++] = '0';
    } else {
        uchar rev[16];
        int rev_len = 0;
        while (temp > 0) {
            rev[rev_len++] = '0' + (temp % 10);
            temp /= 10;
        }
        for (int i = rev_len - 1; i >= 0; i--) {
            nonce_str[len++] = rev[i];
        }
    }
    
    uchar hash[32];
    sha256_transform(nonce_str, len, hash);
    
    // Count leading zeros
    int leading_zeros = 0;
    for (int i = 0; i < 32; i++) {
        uchar byte = hash[i];
        if (byte == 0) {
            leading_zeros += 2;
        } else if ((byte & 0xF0) == 0) {
            leading_zeros += 1;
            break;
        } else {
            break;
        }
    }
    
    // Count total zeros
    int total_zeros = 0;
    for (int i = 0; i < 32; i++) {
        uchar byte = hash[i];
        if ((byte & 0xF0) == 0) total_zeros++;
        if ((byte & 0x0F) == 0) total_zeros++;
    }
    
    // Update stats atomically (parallel-safe)
    atomic_max(&stats_buffer[0], leading_zeros);  // max leading zeros
    atomic_max(&stats_buffer[1], total_zeros);     // max total zeros
    atomic_add(&stats_buffer[2], leading_zeros);   // sum leading zeros
    atomic_add(&stats_buffer[3], total_zeros);     // sum total zeros
    
    // Check if found target
    if (leading_zeros >= target_zeros) {
        if (atomic_cmpxchg(found_flag, 0, 1) == 0) {  // First one to find it
            *found_nonce = nonce;
            for (int i = 0; i < 32; i++) {
                found_hash[i] = hash[i];
            }
        }
    }
}
"""

# Configuration
TARGET_LEADING_ZEROS = 8
BATCH_SIZE = 1024 * 1024 * 4  # 4M hashes per batch - fully parallel
PRINT_INTERVAL = 1  # Print every batch

# Initialize OpenCL
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
prg = cl.Program(ctx, opencl_sha256_kernel).build()
kernel = cl.Kernel(prg, "mine_sha256")

# Persistent buffers (reused across batches for speed)
found_nonce_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 4)
found_hash_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 32)
found_flag_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 4)
stats_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 16)  # 4 uint stats

# Global stats
attempt = 0
global_stats = {
    "max_leading_zeros": 0,
    "max_total_zeros": 0,
    "total_leading_zeros": 0,
    "total_total_zeros": 0,
}

start_time = time.time()
current_nonce = 1
batch_count = 0

print(f"Parallel OpenCL SHA-256 Miner")
print(f"Target: {TARGET_LEADING_ZEROS} leading zeros")
print(f"Batch size: {BATCH_SIZE:,} hashes (fully parallel)")
print(f"Device: {ctx.devices[0].name}")
print("=" * 70)

while True:
    batch_count += 1
    
    # Reset found flag for this batch
    cl.enqueue_fill_buffer(queue, found_flag_buf, np.int32(0), 0, 4)
    cl.enqueue_fill_buffer(queue, stats_buf, np.uint32(0), 0, 16)
    
    # Launch kernel - ALL work happens in parallel on GPU
    kernel(queue, (BATCH_SIZE,), None,
           found_nonce_buf, found_hash_buf, found_flag_buf, stats_buf,
           np.uint32(current_nonce), np.uint32(TARGET_LEADING_ZEROS))
    
    # Only read back minimal data (not all hashes!)
    found_flag = np.empty(1, dtype=np.int32)
    batch_stats = np.empty(4, dtype=np.uint32)
    
    cl.enqueue_copy(queue, found_flag, found_flag_buf)
    cl.enqueue_copy(queue, batch_stats, stats_buf)
    queue.finish()
    
    # Update global stats
    global_stats["max_leading_zeros"] = max(global_stats["max_leading_zeros"], batch_stats[0])
    global_stats["max_total_zeros"] = max(global_stats["max_total_zeros"], batch_stats[1])
    global_stats["total_leading_zeros"] += batch_stats[2]
    global_stats["total_total_zeros"] += batch_stats[3]
    
    attempt += BATCH_SIZE
    current_nonce += BATCH_SIZE
    
    # Print stats
    if batch_count % PRINT_INTERVAL == 0:
        elapsed = time.time() - start_time
        hashrate = attempt / elapsed if elapsed > 0 else 0
        avg_leading = global_stats["total_leading_zeros"] / attempt
        avg_total = global_stats["total_total_zeros"] / attempt
        
        print(f"Batch {batch_count:>4} | Attempts: {attempt:>12,} | Rate: {hashrate:>10,.0f} H/s")
        print(f"         | Max Leading: {global_stats['max_leading_zeros']:>2} | Max Total: {global_stats['max_total_zeros']:>2} | Avg L: {avg_leading:.2f} | Avg T: {avg_total:.2f}")
        print("-" * 70)
    
    # Check if found
    if found_flag[0] == 1:
        # Read back winning data
        found_nonce = np.empty(1, dtype=np.uint32)
        found_hash = np.empty(32, dtype=np.uint8)
        
        cl.enqueue_copy(queue, found_nonce, found_nonce_buf)
        cl.enqueue_copy(queue, found_hash, found_hash_buf)
        queue.finish()
        
        elapsed = time.time() - start_time
        hashrate = attempt / elapsed if elapsed > 0 else 0
        hash_hex = ''.join(f'{b:02x}' for b in found_hash)
        
        # Count zeros for display
        leading_zeros = 0
        for byte in found_hash:
            if byte == 0:
                leading_zeros += 2
            elif (byte & 0xF0) == 0:
                leading_zeros += 1
                break
            else:
                break
        
        total_zeros = hash_hex.count('0')
        
        print("=" * 70)
        print(f"🎉 SUCCESS!")
        print(f"Nonce: {found_nonce[0]}")
        print(f"Hash: {hash_hex}")
        print(f"Leading zeros: {leading_zeros}")
        print(f"Total zeros: {total_zeros}")
        print(f"Attempts: {attempt:,}")
        print(f"Hashrate: {hashrate:,.0f} H/s")
        print(f"Time: {elapsed:.2f} seconds")
        print("=" * 70)
        break
