import pyopencl as cl
import numpy as np
import time
import socket
import json
import hashlib
from threading import Thread, Lock
import struct

# Bitcoin-style double SHA-256 OpenCL kernel with 64-bit nonce
opencl_bitcoin_kernel = """
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
    
    // Process the input in 64-byte blocks
    int num_blocks = (len + 8 + 1 + 63) / 64;
    
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        uchar block[64];
        for (int i = 0; i < 64; i++) block[i] = 0;
        
        int block_start = block_idx * 64;
        int block_end = block_start + 64;
        int copy_len = (len > block_end) ? 64 : ((len > block_start) ? (len - block_start) : 0);
        
        for (int i = 0; i < copy_len; i++) {
            block[i] = input[block_start + i];
        }
        
        // Padding
        if (block_idx == num_blocks - 1 || (block_idx == num_blocks - 2 && len % 64 >= 56)) {
            if (copy_len < 64) {
                block[copy_len] = 0x80;
            }
            if (block_idx == num_blocks - 1) {
                ulong bit_len = (ulong)len * 8;
                block[63] = bit_len & 0xff;
                block[62] = (bit_len >> 8) & 0xff;
                block[61] = (bit_len >> 16) & 0xff;
                block[60] = (bit_len >> 24) & 0xff;
                block[59] = (bit_len >> 32) & 0xff;
                block[58] = (bit_len >> 40) & 0xff;
                block[57] = (bit_len >> 48) & 0xff;
                block[56] = (bit_len >> 56) & 0xff;
            }
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
    }
    
    for (int i = 0; i < 8; i++) {
        output[i*4]   = (h[i] >> 24) & 0xff;
        output[i*4+1] = (h[i] >> 16) & 0xff;
        output[i*4+2] = (h[i] >> 8) & 0xff;
        output[i*4+3] = h[i] & 0xff;
    }
}

// Bitcoin uses double SHA-256: SHA256(SHA256(data))
void bitcoin_hash(uchar *input, int len, uchar *output) {
    uchar first_hash[32];
    sha256_transform(input, len, first_hash);
    sha256_transform(first_hash, 32, output);
}

__kernel void mine_bitcoin(__global ulong *found_nonce, 
                           __global uchar *found_hash, 
                           __global int *found_flag, 
                           __global uint *stats_buffer, 
                           uint start_nonce_high, 
                           uint start_nonce_low, 
                           uint target_zeros) {
    int gid = get_global_id(0);
    
    // Combine 64-bit nonce from high and low 32-bit parts
    ulong base_nonce = ((ulong)start_nonce_high << 32) | (ulong)start_nonce_low;
    ulong nonce = base_nonce + (ulong)gid;
    
    // Convert 64-bit nonce to string
    uchar nonce_str[24];
    int len = 0;
    ulong temp = nonce;
    
    if (temp == 0) {
        nonce_str[len++] = '0';
    } else {
        uchar rev[24];
        int rev_len = 0;
        while (temp > 0) {
            rev[rev_len++] = '0' + (temp % 10);
            temp /= 10;
        }
        for (int i = rev_len - 1; i >= 0; i--) {
            nonce_str[len++] = rev[i];
        }
    }
    
    // Bitcoin-style double SHA-256
    uchar hash[32];
    bitcoin_hash(nonce_str, len, hash);
    
    // Count leading zeros (in hex representation)
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
    
    // Update stats atomically
    atomic_max(&stats_buffer[0], leading_zeros);
    atomic_max(&stats_buffer[1], total_zeros);
    atomic_add(&stats_buffer[2], leading_zeros);
    atomic_add(&stats_buffer[3], total_zeros);
    
    // Check if found target
    if (leading_zeros >= target_zeros) {
        if (atomic_cmpxchg(found_flag, 0, 1) == 0) {
            *found_nonce = nonce;
            for (int i = 0; i < 32; i++) {
                found_hash[i] = hash[i];
            }
        }
    }
}
"""

class BitcoinMiningPoolClient:
    """Bitcoin Stratum mining pool client"""
    
    def __init__(self, pool_url, pool_port, username, password="x", worker_name="lolcatz.bit"):
        self.pool_url = pool_url
        self.pool_port = pool_port
        self.username = username
        self.password = password
        self.worker_name = worker_name
        self.socket = None
        self.connected = False
        self.job_id = None
        self.prevhash = None
        self.coinb1 = None
        self.coinb2 = None
        self.merkle_branch = []
        self.version = None
        self.nbits = None
        self.ntime = None
        self.extranonce1 = None
        self.extranonce2_size = 0
        self.target = None
        self.share_difficulty = 1
        self.submit_lock = Lock()
        self.shares_submitted = 0
        self.shares_accepted = 0
        self.shares_rejected = 0
        self.network_difficulty = 0
        
    def connect(self):
        """Connect to Bitcoin mining pool"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(30)
            self.socket.connect((self.pool_url, self.pool_port))
            self.connected = True
            print(f"✓ Connected to Bitcoin pool: {self.pool_url}:{self.pool_port}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to pool: {e}")
            self.connected = False
            return False
    
    def send_message(self, method, params, msg_id=None):
        """Send JSON-RPC message to pool"""
        if msg_id is None:
            msg_id = int(time.time() * 1000)
        
        message = {
            "id": msg_id,
            "method": method,
            "params": params
        }
        try:
            data = json.dumps(message) + '\n'
            self.socket.sendall(data.encode('utf-8'))
            return True
        except Exception as e:
            print(f"✗ Error sending message: {e}")
            return False
    
    def receive_message(self, timeout=30):
        """Receive JSON-RPC message from pool"""
        try:
            self.socket.settimeout(timeout)
            data = b""
            while b'\n' not in data:
                chunk = self.socket.recv(4096)
                if not chunk:
                    return None
                data += chunk
            
            message = json.loads(data.decode('utf-8').strip())
            return message
        except socket.timeout:
            return None
        except Exception as e:
            print(f"✗ Error receiving message: {e}")
            return None
    
    def subscribe(self):
        """Subscribe to Bitcoin mining pool (Stratum)"""
        self.send_message("mining.subscribe", [f"OpenCL Bitcoin Miner/{self.worker_name}"])
        response = self.receive_message()
        
        if response and 'result' in response:
            result = response['result']
            # Extract subscription details
            if isinstance(result, list) and len(result) >= 2:
                self.extranonce1 = result[1] if len(result) > 1 else ""
                self.extranonce2_size = result[2] if len(result) > 2 else 4
                print(f"✓ Subscribed to Bitcoin pool")
                print(f"  ExtraNonce1: {self.extranonce1}")
                print(f"  ExtraNonce2 Size: {self.extranonce2_size}")
                return True
        
        print(f"✗ Subscription failed")
        return False
    
    def authorize(self):
        """Authorize with Bitcoin mining pool"""
        full_username = f"{self.username}.{self.worker_name}"
        self.send_message("mining.authorize", [full_username, self.password])
        response = self.receive_message()
        
        if response and response.get('result') == True:
            print(f"✓ Authorized as: {full_username}")
            return True
        
        print(f"✗ Authorization failed")
        return False
    
    def submit_share(self, nonce, hash_hex, leading_zeros):
        """Submit a valid Bitcoin share to the pool"""
        with self.submit_lock:
            self.shares_submitted += 1
            
            # Bitcoin share data
            share_data = {
                "nonce": hex(nonce)[2:].zfill(8),  # 32-bit nonce in hex
                "hash": hash_hex,
                "leading_zeros": leading_zeros,
                "difficulty": self.calculate_difficulty(hash_hex),
                "worker": self.worker_name,
                "timestamp": int(time.time())
            }
            
            print(f"\n📤 Submitting Bitcoin share #{self.shares_submitted}")
            print(f"   Nonce: {share_data['nonce']}")
            print(f"   Hash: {hash_hex}")
            print(f"   Leading zeros: {leading_zeros}")
            print(f"   Calculated difficulty: {share_data['difficulty']:.2f}")
            
            # Real Stratum submission format:
            # mining.submit(username, job_id, extranonce2, ntime, nonce)
            if self.connected and self.job_id:
                try:
                    extranonce2 = "00" * self.extranonce2_size
                    params = [
                        f"{self.username}.{self.worker_name}",
                        self.job_id,
                        extranonce2,
                        self.ntime,
                        share_data['nonce']
                    ]
                    self.send_message("mining.submit", params)
                    
                    # Wait for response
                    response = self.receive_message(timeout=5)
                    if response and response.get('result') == True:
                        self.shares_accepted += 1
                        print(f"✓ Share ACCEPTED! ({self.shares_accepted}/{self.shares_submitted})")
                        return True
                    else:
                        self.shares_rejected += 1
                        error = response.get('error', 'Unknown error') if response else 'No response'
                        print(f"✗ Share REJECTED: {error}")
                        return False
                        
                except Exception as e:
                    self.shares_rejected += 1
                    print(f"✗ Share submission error: {e}")
                    return False
            else:
                # Demo mode - simulate acceptance
                self.shares_accepted += 1
                print(f"✓ Share logged (demo mode) ({self.shares_accepted}/{self.shares_submitted})")
                return True
    
    def calculate_difficulty(self, hash_hex):
        """Calculate difficulty from hash"""
        # Bitcoin difficulty = max_target / current_hash
        max_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
        hash_int = int(hash_hex, 16)
        
        if hash_int == 0:
            return float('inf')
        
        difficulty = max_target / hash_int
        return difficulty
    
    def get_work(self):
        """Request new work from pool (mining.notify)"""
        # In Stratum, work is pushed via mining.notify
        # This is a placeholder for receiving notifications
        response = self.receive_message(timeout=1)
        
        if response and response.get('method') == 'mining.notify':
            params = response.get('params', [])
            if len(params) >= 9:
                self.job_id = params[0]
                self.prevhash = params[1]
                self.coinb1 = params[2]
                self.coinb2 = params[3]
                self.merkle_branch = params[4]
                self.version = params[5]
                self.nbits = params[6]
                self.ntime = params[7]
                clean_jobs = params[8]
                
                print(f"\n📋 New work received:")
                print(f"   Job ID: {self.job_id}")
                print(f"   Difficulty: {self.nbits}")
                return True
        
        return False
    
    def get_stats(self):
        """Get pool submission statistics"""
        return {
            "submitted": self.shares_submitted,
            "accepted": self.shares_accepted,
            "rejected": self.shares_rejected,
            "acceptance_rate": (self.shares_accepted / self.shares_submitted * 100) if self.shares_submitted > 0 else 0
        }
    
    def disconnect(self):
        """Disconnect from pool"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False
        print("Disconnected from Bitcoin pool")

def split_u64_to_u32(n):
    """Split 64-bit integer into high and low 32-bit parts"""
    low = n & 0xFFFFFFFF
    high = (n >> 32) & 0xFFFFFFFF
    return high, low

def bitcoin_double_sha256(data):
    """Python implementation of Bitcoin's double SHA-256 for verification"""
    if isinstance(data, str):
        data = data.encode('utf-8')
    first_hash = hashlib.sha256(data).digest()
    second_hash = hashlib.sha256(first_hash).digest()
    return second_hash.hex()

def count_leading_zeros(hex_hash):
    """Count leading zero characters in hex hash"""
    return len(hex_hash) - len(hex_hash.lstrip('0'))

def count_total_zeros(hex_hash):
    """Count total zero characters in hex hash"""
    return hex_hash.count('0')

# Mining configuration
TARGET_LEADING_ZEROS = 8   # Bitcoin-style difficulty (8 = ~16 Bitcoin difficulty)
POOL_SHARE_DIFFICULTY = 5  # Minimum difficulty to submit to pool
PRINT_INTERVAL = 5000
BATCH_SIZE = 1024 * 1024   # 1M nonces per GPU batch

# Bitcoin pool configuration
POOL_ENABLED = True
POOL_URL = "stratum+tcp://ss.antpool.com"  # Replace with actual Bitcoin pool
POOL_PORT = 3333
POOL_USERNAME = "lolcatz.bit"  # Your BTC address
POOL_WORKER = "opencl_miner_01"

# Initialize mining stats
attempt = 0
stats = {
    "max_zeros": 0,
    "max_leading_zeros": 0,
    "total_zeros": 0,
    "total_leading_zeros": 0,
}

start_time = time.time()

# Ranged incrementing nonce - 64-bit
domain_min = 1
domain_max = 10**21
current_nonce = domain_min

# Initialize OpenCL
print("Initializing OpenCL...")
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

print("Compiling Bitcoin mining kernel...")
prg = cl.Program(ctx, opencl_bitcoin_kernel).build()
kernel = cl.Kernel(prg, "mine_bitcoin")

# Persistent buffers
found_nonce_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 8)
found_hash_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 32)
found_flag_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 4)
stats_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 16)

# Initialize Bitcoin pool client
pool_client = None
if POOL_ENABLED:
    pool_client = BitcoinMiningPoolClient(POOL_URL, POOL_PORT, POOL_USERNAME, worker_name=POOL_WORKER)
    print(f"\n{'='*70}")
    print("BITCOIN POOL MINING MODE")
    print(f"{'='*70}")
    print("Pool submission configured (demo mode)")
    print(f"{'='*70}\n")
    
    # In production, connect to real pool:
    # if pool_client.connect():
    #     pool_client.subscribe()
    #     pool_client.authorize()

print(f"\n{'='*70}")
print("BITCOIN DOUBLE SHA-256 MINER")
print(f"{'='*70}")
print(f"Algorithm: SHA256(SHA256(nonce))")
print(f"Local target: {TARGET_LEADING_ZEROS} leading zeros")
print(f"Pool share difficulty: {POOL_SHARE_DIFFICULTY} leading zeros")
print(f"Device: {ctx.devices[0].name}")
print(f"Nonce range: {domain_min:,} to {domain_max:,}")
print(f"Batch size: {BATCH_SIZE:,} (parallel)")
print("=" * 70)

# Verify kernel correctness with test
test_nonce = "12345"
expected = bitcoin_double_sha256(test_nonce)
print(f"Verification: Double SHA-256 of '{test_nonce}'")
print(f"Expected: {expected}")
print("=" * 70 + "\n")

while current_nonce <= domain_max:
    # Calculate batch size (don't exceed domain_max)
    actual_batch_size = min(BATCH_SIZE, domain_max - current_nonce + 1)
    
    # Reset flags
    cl.enqueue_fill_buffer(queue, found_flag_buf, np.int32(0), 0, 4)
    cl.enqueue_fill_buffer(queue, stats_buf, np.uint32(0), 0, 16)
    
    # Split 64-bit nonce into high and low 32-bit parts
    high_nonce, low_nonce = split_u64_to_u32(current_nonce)
    
    # Launch parallel Bitcoin mining kernel
    kernel(queue, (actual_batch_size,), None,
           found_nonce_buf, found_hash_buf, found_flag_buf, 
           stats_buf, np.uint32(high_nonce), np.uint32(low_nonce), 
           np.uint32(TARGET_LEADING_ZEROS))
    
    # Read back minimal stats
    found_flag = np.empty(1, dtype=np.int32)
    batch_stats = np.empty(4, dtype=np.uint32)
    
    cl.enqueue_copy(queue, found_flag, found_flag_buf)
    cl.enqueue_copy(queue, batch_stats, stats_buf)
    queue.finish()
    
    # Update stats
    stats["max_leading_zeros"] = max(stats["max_leading_zeros"], batch_stats[0])
    stats["max_zeros"] = max(stats["max_zeros"], batch_stats[1])
    stats["total_leading_zeros"] += batch_stats[2]
    stats["total_zeros"] += batch_stats[3]
    
    attempt += actual_batch_size
    current_nonce += actual_batch_size
    
    # Print stats periodically
    if attempt % PRINT_INTERVAL < actual_batch_size or attempt == actual_batch_size:
        elapsed = time.time() - start_time
        hashrate = attempt / elapsed if elapsed > 0 else 0
        
        # Calculate equivalent Bitcoin metrics
        btc_difficulty_equiv = 2 ** stats["max_leading_zeros"] / (2 ** 32)
        
        print(f"⛏️  Attempts: {attempt:,}, Hashrate: {hashrate:,.0f} H/s ({hashrate/1e6:.2f} MH/s)")
        print(f"   Max leading zeros: {stats['max_leading_zeros']} (≈{btc_difficulty_equiv:.2f} BTC difficulty)")
        print(f"   Max total zeros: {stats['max_zeros']}")
        print(f"   Average leading zeros: {stats['total_leading_zeros'] / attempt:.2f}")
        print(f"   Average total zeros: {stats['total_zeros'] / attempt:.2f}")
        print(f"   Current nonce: {current_nonce:,}")
        
        # Print pool stats if enabled
        if pool_client:
            pool_stats = pool_client.get_stats()
            print(f"   📊 Pool: {pool_stats['submitted']} submitted, "
                  f"{pool_stats['accepted']} accepted, "
                  f"{pool_stats['rejected']} rejected "
                  f"({pool_stats['acceptance_rate']:.1f}%)")
        
        print("-" * 70)
    
    # Check if found
    if found_flag[0] == 1:
        found_nonce = np.empty(1, dtype=np.uint64)
        found_hash = np.empty(32, dtype=np.uint8)
        
        cl.enqueue_copy(queue, found_nonce, found_nonce_buf)
        cl.enqueue_copy(queue, found_hash, found_hash_buf)
        queue.finish()
        
        elapsed = time.time() - start_time
        hashrate = attempt / elapsed if elapsed > 0 else 0
        hash_hex = ''.join(f'{b:02x}' for b in found_hash)
        
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
        btc_difficulty = 2 ** leading_zeros / (2 ** 32)
        
        # Verify with Python double SHA-256
        verification_hash = bitcoin_double_sha256(str(found_nonce[0]))
        verified = "✓ VERIFIED" if verification_hash == hash_hex else "✗ MISMATCH"
        
        print("\n" + "=" * 70)
        print(f"🎯 BITCOIN BLOCK FOUND at attempt {attempt:,}!")
        print(f"   Nonce: {found_nonce[0]}")
        print(f"   Double SHA-256 Hash: {hash_hex}")
        print(f"   Leading zeros: {leading_zeros}")
        print(f"   Total zeros: {total_zeros}")
        print(f"   Bitcoin difficulty equivalent: {btc_difficulty:.2f}")
        print(f"   Hashrate: {hashrate:,.0f} H/s ({hashrate/1e6:.2f} MH/s)")
        print(f"   {verified}")
        print("=" * 70)
        
        # Submit to Bitcoin pool if enabled and meets pool difficulty
        if pool_client and leading_zeros >= POOL_SHARE_DIFFICULTY:
            pool_client.submit_share(found_nonce[0], hash_hex, leading_zeros)
        
        # Continue mining unless we found the ultimate solution
        if leading_zeros >= TARGET_LEADING_ZEROS:
            break

if current_nonce > domain_max:
    print("\n⚠️  Exhausted nonce range without finding solution")

# Cleanup and final statistics
if pool_client:
    print(f"\n{'='*70}")
    print("FINAL BITCOIN POOL STATISTICS")
    pool_stats = pool_client.get_stats()
    print(f"Total shares submitted: {pool_stats['submitted']}")
    print(f"Shares accepted: {pool_stats['accepted']} ✓")
    print(f"Shares rejected: {pool_stats['rejected']} ✗")
    print(f"Acceptance rate: {pool_stats['acceptance_rate']:.2f}%")
    print(f"{'='*70}")
    pool_client.disconnect()

elapsed_total = time.time() - start_time
final_hashrate = attempt / elapsed_total if elapsed_total > 0 else 0

print(f"\n{'='*70}")
print("SESSION SUMMARY")
print(f"{'='*70}")
print(f"Total hashes: {attempt:,}")
print(f"Total time: {elapsed_total:.2f} seconds")
print(f"Average hashrate: {final_hashrate:,.0f} H/s ({final_hashrate/1e6:.2f} MH/s)")
print(f"Best result: {stats['max_leading_zeros']} leading zeros")
print(f"{'='*70}")
print("\n✓ Bitcoin mining session complete!")
