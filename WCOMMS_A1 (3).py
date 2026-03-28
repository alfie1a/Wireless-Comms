import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# BINARY AMPLITUDE SHIFT KEYING (BASK)
# ==============================================================================
def bask_mod(bits):
    """
    BASK Modulation - Custom Implementation
    Bit 0 -> 0+0j, Bit 1 -> 1+0j
    """
    bits = np.array(bits)
    sym = np.zeros(len(bits), dtype=complex)
    
    for i, b in enumerate(bits):
        sym[i] = 1 + 0j if b == 1 else 0 + 0j
    
    return sym


def bask_demod(rx_sym):
    """
    BASK Demodulation using Euclidean Distance
    Finds nearest constellation point
    """
    const = np.array([0 + 0j, 1 + 0j])
    bits_out = []
    
    for s in rx_sym:
        # Calculate distance to each point
        d = np.abs(s - const)
        # Choose minimum distance
        bits_out.append(np.argmin(d))
    
    return np.array(bits_out)


# ==============================================================================
# QUADRATURE PHASE SHIFT KEYING (QPSK)
# ==============================================================================
def qpsk_mod(bits):
    
    bits = np.array(bits)
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)
    
    pairs = bits.reshape(-1, 2)
    sym = np.zeros(len(pairs), dtype=complex)
    
    # Gray-coded mapping
    n = 1 / np.sqrt(2)
    for i, p in enumerate(pairs):
        if p[0] == 0 and p[1] == 0:
            sym[i] = (-1 - 1j) * n      # 00 -> 45 deg
        elif p[0] == 0 and p[1] == 1:
            sym[i] = (-1 + 1j) * n     # 01 -> 135 deg
        elif p[0] == 1 and p[1] == 1:
            sym[i] = (1 + 1j) * n     # 11 -> 225 deg
        else:
            sym[i] = (1 - 1j) * n      # 10 -> 315 deg
    
    return sym


def qpsk_demod(rx_sym):
    """
    QPSK Demodulation using Euclidean Distance
    Returns demodulated bit pairs
    """
    # Constellation points
    n = 1 / np.sqrt(2)
    const = np.array([
        (-1 - 1j) * n,   # 00
        (-1 + 1j) * n,  # 01
        (1 + 1j) * n,  # 11
        (1 - 1j) * n    # 10
    ])
    
    map_bits = [[0, 0], [0, 1], [1, 1], [1, 0]]
    bits_out = np.zeros(len(rx_sym) * 2, dtype=int)
    
    for i, s in enumerate(rx_sym):
        # Find minimum distance
        d = np.abs(s - const)
        idx = np.argmin(d)
        bits_out[2*i:2*i+2] = map_bits[idx]
    
    return bits_out


# ==============================================================================
# 8-PHASE SHIFT KEYING (8-PSK)
# ==============================================================================
def psk8_mod(bits):

    bits = np.array(bits)
    rem = len(bits) % 3
    if rem != 0:
        bits = np.append(bits, np.zeros(3 - rem, dtype=int))
    
    triples = bits.reshape(-1, 3)
    sym = np.zeros(len(triples), dtype=complex)
    
    # Gray code mapping
    gray = {
        (0, 0, 0): 0, (0, 0, 1): 1, (0, 1, 1): 2, (0, 1, 0): 3,
        (1, 1, 0): 4, (1, 1, 1): 5, (1, 0, 1): 6, (1, 0, 0): 7
    }
    
    for i, t in enumerate(triples):
        k = tuple(t)
        p = gray[k]
        angle = p * np.pi / 4
        sym[i] = np.exp(1j * angle)
    
    return sym


def psk8_demod(rx_sym):
    """
    8-PSK Demodulation using Euclidean Distance
    Returns demodulated bit triplets
    """
    # Generate constellation
    const = np.array([np.exp(1j * k * np.pi / 4) for k in range(8)])
    
    # Bit mappings
    map_bits = [
        [0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0],
        [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]
    ]
    
    bits_out = np.zeros(len(rx_sym) * 3, dtype=int)
    
    for i, s in enumerate(rx_sym):
        # Find nearest point
        d = np.abs(s - const)
        idx = np.argmin(d)
        bits_out[3*i:3*i+3] = map_bits[idx]
    
    return bits_out


# ==============================================================================
# AWGN CHANNEL
# ==============================================================================
def add_noise(sym, eb_n0_db, k):
    """
    Add AWGN to symbols
    eb_n0_db: Eb/N0 in dB
    k: bits per symbol
    """
    snr = 10 ** (eb_n0_db / 10)
    var = 1 / (2 * snr * k)
    
    if np.iscomplexobj(sym):
        n_i = np.random.normal(0, np.sqrt(var), len(sym))
        n_q = np.random.normal(0, np.sqrt(var), len(sym))
        noise = n_i + 1j * n_q
    else:
        noise = np.random.normal(0, np.sqrt(2 * var), len(sym))
    
    return sym + noise


def calc_ber(tx, rx):
    """
    Calculate BER
    """
    L = min(len(tx), len(rx))
    errs = np.sum(tx[:L] != rx[:L])
    return errs / L


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================
def plot_const(sym, title, pos=None, color='blue'):

    if pos:
        plt.subplot(*pos)
    
    if np.iscomplexobj(sym):
        plt.scatter(np.real(sym), np.imag(sym), alpha=0.7, s=60, color=color, edgecolors='black', linewidth=0.5)
        plt.xlabel('In-Phase (I)')
        plt.ylabel('Quadrature (Q)')
    else:
        plt.scatter(sym, np.zeros_like(sym), alpha=0.7, s=60, color=color, edgecolors='black', linewidth=0.5)
        plt.xlabel('Amplitude')
        plt.ylabel('Quadrature (always 0)')
    
    plt.title(title, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')


def plot_ber(snr, ber1, ber2, ber3):
    """
    Plot BER curves
    """
    plt.figure(figsize=(12, 8))
    
    
    plt.semilogy(snr, ber1, 'bo-', label='BASK', linewidth=2, markersize=6, markerfacecolor='lightblue')
    plt.semilogy(snr, ber2, 'rs-', label='QPSK', linewidth=2, markersize=6, markerfacecolor='lightcoral')
    plt.semilogy(snr, ber3, 'g^-', label='8-PSK', linewidth=2, markersize=6, markerfacecolor='lightgreen')
    
    # Highlight 10 dB
    idx = np.where(snr == 10)[0]
    if len(idx) > 0:
        j = idx[0]
        plt.semilogy(snr[j], ber1[j], 'ko', markersize=15, 
                     markerfacecolor='yellow', markeredgewidth=2, label='Eb/N0=10dB')
        plt.semilogy(snr[j], ber2[j], 'ko', markersize=15, 
                     markerfacecolor='yellow', markeredgewidth=2)
        plt.semilogy(snr[j], ber3[j], 'ko', markersize=15, 
                     markerfacecolor='yellow', markeredgewidth=2)
        plt.axvline(x=10, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.xlabel('Eb/N0 (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('BER Performance Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=11)
    plt.xlim([snr[0], snr[-1]])
    plt.ylim([1e-6, 1])
    plt.tight_layout()


# ==============================================================================
# MAIN PROGRAM
# ==============================================================================
if __name__ == "__main__":
    
    
    # Test bits
    test = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1])
    
    print("=" * 60)
    print("DIGITAL MODULATION SIMULATION")
    print("=" * 60)
    print(f"Test bits: {test}\n")
    
    # ==============================================================================
    # BASK TEST
    # ==============================================================================
    print("1. BINARY AMPLITUDE SHIFT KEYING (BASK)")
    print("-" * 60)
    
    tx1 = bask_mod(test)
    print(f"TX: {tx1}")
    
    rx1 = add_noise(tx1, 10, 1)
    dec1 = bask_demod(rx1)
    print(f"RX: {dec1}")
    print(f"Errors: {np.sum(test != dec1)}\n")
    
    # ==============================================================================
    # QPSK TEST
    # ==============================================================================
    print("2. QUADRATURE PHASE SHIFT KEYING (QPSK)")
    print("-" * 60)
    
    tx2 = qpsk_mod(test)
    print(f"TX: {tx2}")
    
    rx2 = add_noise(tx2, 10, 2)
    dec2 = qpsk_demod(rx2)
    print(f"RX: {dec2}")
    print(f"Errors: {np.sum(test != dec2[:len(test)])}\n")
    
    # ==============================================================================
    # 8-PSK TEST
    # ==============================================================================
    print("3. 8-PHASE SHIFT KEYING (8-PSK)")
    print("-" * 60)
    
    tx3 = psk8_mod(test)
    print(f"TX: {tx3}")
    
    rx3 = add_noise(tx3, 10, 3)
    dec3 = psk8_demod(rx3)
    print(f"RX: {dec3}")
    print(f"Errors: {np.sum(test != dec3[:len(test)])}\n")
    
    # ==============================================================================
    #  TRANSMIT POSSIBILITIES CONSTELLATIONS - 
    # ==============================================================================
    print("Generating constellation diagrams...")
    
    plt.figure(figsize=(15, 5))
    
    # BASK - Royal Blue
    s1 = bask_mod([0, 1])
    plot_const(s1, 'BASK - TRANSMIT POSSIBILITIES Constellation', (1, 3, 1), color='royalblue')
    
    # QPSK - Crimson Red
    s2 = qpsk_mod([1, 0, 0, 0, 1, 1, 0, 1])
    plot_const(s2, 'QPSK - TRANSMIT POSSIBILITIES Constellation', (1, 3, 2), color='crimson')
    
    # 8-PSK - Forest Green
    s3 = psk8_mod([1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0])
    plot_const(s3, '8-PSK - TRANSMIT POSSIBILITIES Constellation', (1, 3, 3), color='forestgreen')
    
    plt.tight_layout()
    plt.savefig('transmit_constellation_diagrams.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(" Saved: transmit_constellation_diagrams.png")
    
    # ==============================================================================
    # NOISY CONSTELLATIONS (20 BITS) 
    # ==============================================================================
    test20 = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1])
    
    plt.figure(figsize=(15, 5))
    
    # BASK - Dark Orange
    tx1_20 = bask_mod(test20)
    rx1_20 = add_noise(tx1_20, 10, 1)
    plot_const(rx1_20, 'BASK - Received (20 bits, Eb/N0=10 dB)', (1, 3, 1), color='darkorange')
    
    # QPSK - Purple
    tx2_20 = qpsk_mod(test20)
    rx2_20 = add_noise(tx2_20, 10, 2)
    plot_const(rx2_20, 'QPSK - Received (20 bits, Eb/N0=10 dB)', (1, 3, 2), color='purple')
    
    # 8-PSK - Teal
    tx3_20 = psk8_mod(test20)
    rx3_20 = add_noise(tx3_20, 10, 3)
    plot_const(rx3_20, '8-PSK - Received (20 bits, Eb/N0=10 dB)', (1, 3, 3), color='teal')
    
    plt.tight_layout()
    plt.savefig('noisy_received_constellations.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(" Saved: noisy_received_constellations.png")
    
    # ==============================================================================
    # BER SIMULATION
    # ==============================================================================
    print("\nRunning BER simulation (Eb/N0 = 0 to 20 dB)...")
    print("This may take a moment...")
    print("=" * 60)
    
    snr_range = np.arange(0, 21, 2)
    N = 50000  # bits per point
    
    ber_bask = []
    ber_qpsk = []
    ber_8psk = []
    
    for snr in snr_range:
        # Random bits
        data = np.random.randint(0, 2, N)
        
        # BASK
        t1 = bask_mod(data)
        r1 = add_noise(t1, snr, 1)
        d1 = bask_demod(r1)
        ber_bask.append(calc_ber(data, d1))
        
        # QPSK
        t2 = qpsk_mod(data)
        r2 = add_noise(t2, snr, 2)
        d2 = qpsk_demod(r2)
        ber_qpsk.append(calc_ber(data, d2[:len(data)]))
        
        # 8-PSK
        t3 = psk8_mod(data)
        r3 = add_noise(t3, snr, 3)
        d3 = psk8_demod(r3)
        ber_8psk.append(calc_ber(data, d3[:len(data)]))
        
        print(f"Eb/N0 = {snr:2d} dB | BASK={ber_bask[-1]:.2e} | "
              f"QPSK={ber_qpsk[-1]:.2e} | 8-PSK={ber_8psk[-1]:.2e}")
    
    print("=" * 60)
    
    # Plot BER 
    plot_ber(snr_range, ber_bask, ber_qpsk, ber_8psk)
    plt.savefig('ber_performance_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(" Saved: ber_performance_curves.png")
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print("Generated files:")
    print("  1. transmit_constellation_diagrams.png")
    print("  2. noisy_received_constellations.png")
    print("  3. ber_performance_curves.png")
    print("\nAll modulation schemes implemented using:")
    print("   Custom modulation functions (no built-in functions)")
    print("   Minimum Euclidean distance demodulation")
    print("   AWGN channel simulation")
    print("   BER performance from 0-20 dB with highlighted 10 dB point")
    print("=" * 60)