import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
from scipy.special import erfc

# parameters 
fft_size = 512
num_active = 480
cp_len = fft_size // 8
subcarrier_spacing = 15e3
sample_rate = 7.68e6
snr_db_range = np.arange(0, 21, 2)

bits_per_sym = 2       # QPSK
num_symbols = 100      


null_fft = (fft_size - num_active) // 2. # Will be used for OFDM mapping



def q_function(x):
    return 0.5 * erfc(x / np.sqrt(2))


def theoretical_ber_qpsk(ebn0_db):
    ebn0_linear = 10 ** (ebn0_db / 10.0)
    return q_function(np.sqrt(2 * ebn0_linear))


# QPSK 
def qpsk_modulate(bits):
    bits = bits.reshape(-1, 2)
    mapping = {
        (0, 0): -1 - 1j,
        (0, 1): -1 + 1j,
        (1, 1):  1 + 1j,
        (1, 0):  1 - 1j
    }
    syms = np.array([mapping[tuple(b)] for b in bits])
    return syms / np.sqrt(2)


def qpsk_demodulate(symbols):
    out = []
    for s in symbols:
        if s.real < 0 and s.imag < 0:
            out.extend([0, 0])
        elif s.real < 0 and s.imag >= 0:
            out.extend([0, 1])
        elif s.real >= 0 and s.imag >= 0:
            out.extend([1, 1])
        else:
            out.extend([1, 0])
    return np.array(out, dtype=int)


# OFDM mapping / TX / RX 
def map_ofdm_single_user(sym_matrix):
    out = np.zeros((sym_matrix.shape[0], fft_size), dtype=complex)
    out[:, null_fft:null_fft + num_active] = sym_matrix
    return out


def demap_ofdm_single_user(freq_grid):
    return freq_grid[:, null_fft:null_fft + num_active]


def ofdm_tx_single_user(syms):
    freq_grid = map_ofdm_single_user(syms)
    time_no_cp = ifft(freq_grid, axis=1) * np.sqrt(fft_size)
    cp = time_no_cp[:, -cp_len:]
    return np.concatenate([cp, time_no_cp], axis=1).reshape(-1)


def ofdm_rx_single_user(rx):
    frame_len = fft_size + cp_len
    frames = rx.reshape(-1, frame_len)
    time_no_cp = frames[:, cp_len:]
    freq_grid = fft(time_no_cp / np.sqrt(fft_size), axis=1)
    return demap_ofdm_single_user(freq_grid)


# AWGN 
def add_awgn(sig, snr_db):
    snr_linear = 10 ** (snr_db / 10.0)
    power_sig = np.mean(np.abs(sig) ** 2)
    noise_power = power_sig / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape)
    )
    return sig + noise


# Part A 
def simulate_ofdm_qpsk():
    bits_per_ofdm = num_active * bits_per_sym
    total_bits = bits_per_ofdm * num_symbols

    ber_sim = []
    ber_th = []
    const_pts = None
    const_ebn0 = 10

    for snr_db in snr_db_range:
        tx_bits = np.random.randint(0, 2, total_bits)
        tx_syms = qpsk_modulate(tx_bits).reshape(num_symbols, num_active)

        tx = ofdm_tx_single_user(tx_syms)
        rx = add_awgn(tx, snr_db)

        rx_grid = ofdm_rx_single_user(rx)       # [num_symbols, num_active]
        rx_syms = rx_grid.reshape(-1)

        rx_bits = qpsk_demodulate(rx_syms)
        ber = np.mean(tx_bits != rx_bits)
        ber_sim.append(ber)
        ber_th.append(theoretical_ber_qpsk(snr_db))

        print(f"[OFDM] Eb/N0={snr_db} dB, BER={ber:.3e}")

        if snr_db == const_ebn0:
            one_symbol = rx_grid[0, :]
            n_plot = min(400, num_active)
            idx = np.random.choice(num_active, size=n_plot, replace=False)
            const_pts = one_symbol[idx].copy()

    return snr_db_range, np.array(ber_sim), np.array(ber_th), const_pts, const_ebn0


def plot_part_a(snr, ber_sim, ber_th, const_pts, const_ebn0):
    plt.figure(figsize=(8, 5))
    plt.semilogy(snr, ber_sim, 'bo-', label='Simulated')
    plt.semilogy(snr, ber_th, 'rx--', label='Theory')
    plt.grid(True, which='both')
    plt.title("Part A: OFDM QPSK BER")
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BER")
    plt.ylim(1e-7, 1)
    plt.legend()
    plt.tight_layout()

    if const_pts is not None:
        plt.figure(figsize=(5, 5))
        plt.scatter(const_pts.real, const_pts.imag,
                    s=25, alpha=0.5,
                    label=f"Received (Eb/N0 = {const_ebn0} dB)")

        ideal_bits = np.array([0, 0, 0, 1, 1, 1, 1, 0]).reshape(-1, 2)
        ideal_syms = qpsk_modulate(ideal_bits.reshape(-1))
        plt.scatter(ideal_syms.real, ideal_syms.imag,
                    marker='x', s=120, label="Ideal QPSK points")

        plt.axhline(0)
        plt.axvline(0)
        plt.xlim([-1.2, 1.2])
        plt.ylim([-1.2, 1.2])
        plt.grid(True)
        plt.title(f"QPSK RX Constellation at Eb/N0 = {const_ebn0} dB")
        plt.xlabel("In-phase (I)")
        plt.ylabel("Quadrature (Q)")
        plt.axis('equal')
        plt.legend()
        plt.tight_layout()


# Part B 
user_sc = num_active // 2


def map_ofdma_two_users(u1, u2):
    out = np.zeros((u1.shape[0], fft_size), dtype=complex)
    a = null_fft
    b = a + user_sc
    c = b + user_sc
    out[:, a:b] = u1
    out[:, b:c] = u2
    return out


def demap_ofdma_two_users(freq):
    a = null_fft
    b = a + user_sc
    c = b + user_sc
    return freq[:, a:b], freq[:, b:c]


def ofdma_tx(u1, u2):
    freq = map_ofdma_two_users(u1, u2)
    time_no_cp = ifft(freq, axis=1) * np.sqrt(fft_size)
    cp = time_no_cp[:, -cp_len:]
    return np.concatenate([cp, time_no_cp], axis=1).reshape(-1)


def ofdma_rx(rx):
    frames = rx.reshape(-1, fft_size + cp_len)
    time_no_cp = frames[:, cp_len:]
    freq = fft(time_no_cp / np.sqrt(fft_size), axis=1)
    return demap_ofdma_two_users(freq)


def simulate_ofdma_two_users():
    bits_user = user_sc * bits_per_sym * num_symbols
    ber1_list = []
    ber2_list = []

    for snr_db in snr_db_range:
        b1 = np.random.randint(0, 2, bits_user)
        b2 = np.random.randint(0, 2, bits_user)

        s1 = qpsk_modulate(b1).reshape(num_symbols, user_sc)
        s2 = qpsk_modulate(b2).reshape(num_symbols, user_sc)

        tx = ofdma_tx(s1, s2)
        rx = add_awgn(tx, snr_db)

        r1, r2 = ofdma_rx(rx)
        r1_bits = qpsk_demodulate(r1.reshape(-1))
        r2_bits = qpsk_demodulate(r2.reshape(-1))

        ber1 = np.mean(b1 != r1_bits)
        ber2 = np.mean(b2 != r2_bits)

        ber1_list.append(ber1)
        ber2_list.append(ber2)

        print(f"[OFDMA] Eb/N0={snr_db} dB, U1={ber1:.3e}, U2={ber2:.3e}")

    return snr_db_range, np.array(ber1_list), np.array(ber2_list)


def plot_part_b(snr, ber1, ber2):
    theory = theoretical_ber_qpsk(snr)

    plt.figure(figsize=(8, 5))
    plt.semilogy(snr, ber1, 'bo-', label='User 1 Sim')
    plt.semilogy(snr, theory, 'b--', label='User 1 Theory')
    plt.grid(True, which='both')
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BER")
    plt.title("Part B: OFDMA BER – User 1")
    plt.ylim(1e-7, 1)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    plt.semilogy(snr, ber2, 'rs-', label='User 2 Sim')
    plt.semilogy(snr, theory, 'r--', label='User 2 Theory')
    plt.grid(True, which='both')
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BER")
    plt.title("Part B: OFDMA BER – User 2")
    plt.ylim(1e-7, 1)
    plt.legend()
    plt.tight_layout()


if __name__ == "__main__":
    snr_a, ber_sim_a, ber_th_a, const_pts, const_ebn0 = simulate_ofdm_qpsk()
    plot_part_a(snr_a, ber_sim_a, ber_th_a, const_pts, const_ebn0)

    snr_b, ber_u1, ber_u2 = simulate_ofdma_two_users()
    plot_part_b(snr_b, ber_u1, ber_u2)

    
    plt.show()


