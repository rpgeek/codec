# -*- coding: utf-8 -*-
"""
Created on 21 Aug 2014

Functions and classes for convolutional and turbo codes.

Author: Venkat Venkatesan
"""


import math
import numpy as np


def bitxor(num):
    '''
    Returns the XOR of the bits in the binary representation of the
    nonnegative integer num.
    '''

    count_of_ones = 0
    while num > 0:
        count_of_ones += num & 1
        num >>= 1

    return count_of_ones % 2


def maxstar(eggs, spam, max_log=False):
    '''
    Returns log(exp(eggs) + exp(spam)) if not max_log, and max(eggs, spam)
    otherwise.
    '''

    return max(eggs, spam) + (
        0 if max_log else math.log(1 + math.exp(-abs(spam - eggs))))


def turbo_int_3gpp2(n_bits):
    '''
    Computes 3GPP2 turbo interleaver and deinterleaver for specified
    packet size.

    Parameters
    ----------
    n_bits : int
        Number of bits per packet. Must exceed 128 and not exceed 32768.

    Returns
    -------
    turbo_int : list
        List of length n_bits with int elements, specifying the
        interleaver.
    turbo_deint : list
        List of length n_bits with int elements, specifying the
        deinterleaver.
    '''

    # Look-up table
    int_table = (
        (1, 1, 3, 5, 1, 5, 1, 5,
         3, 5, 3, 5, 3, 5, 5, 1,
         3, 5, 3, 5, 3, 5, 5, 5,
         1, 5, 1, 5, 3, 5, 5, 3),
        (5, 15, 5, 15, 1, 9, 9, 15,
         13, 15, 7, 11, 15, 3, 15, 5,
         13, 15, 9, 3, 1, 3, 15, 1,
         13, 1, 9, 15, 11, 3, 15, 5),
        (27, 3, 1, 15, 13, 17, 23, 13,
         9, 3, 15, 3, 13, 1, 13, 29,
         21, 19, 1, 3, 29, 17, 25, 29,
         9, 13, 23, 13, 13, 1, 13, 13),
        (3, 27, 15, 13, 29, 5, 1, 31,
         3, 9, 15, 31, 17, 5, 39, 1,
         19, 27, 15, 13, 45, 5, 33, 15,
         13, 9, 15, 31, 17, 5, 15, 33),
        (15, 127, 89, 1, 31, 15, 61, 47,
         127, 17, 119, 15, 57, 123, 95, 5,
         85, 17, 55, 57, 15, 41, 93, 87,
         63, 15, 13, 15, 81, 57, 31, 69),
        (3, 1, 5, 83, 19, 179, 19, 99,
         23, 1, 3, 13, 13, 3, 17, 1,
         63, 131, 17, 131, 211, 173, 231, 171,
         23, 147, 243, 213, 189, 51, 15, 67),
        (13, 335, 87, 15, 15, 1, 333, 11,
         13, 1, 121, 155, 1, 175, 421, 5,
         509, 215, 47, 425, 295, 229, 427, 83,
         409, 387, 193, 57, 501, 313, 489, 391),
        (1, 349, 303, 721, 973, 703, 761, 327,
         453, 95, 241, 187, 497, 909, 769, 349,
         71, 557, 197, 499, 409, 259, 335, 253,
         677, 717, 313, 757, 189, 15, 75, 163))

    # Results of bit-reversing 0,1,...,31
    bit_rev32 = (
        0, 16, 8, 24, 4, 20, 12, 28,
        2, 18, 10, 26, 6, 22, 14, 30,
        1, 17, 9, 25, 5, 21, 13, 29,
        3, 19, 11, 27, 7, 23, 15, 31)

    # Integer n such that 3 <= n <= 10
    interleaver_param = math.ceil(math.log(n_bits) / math.log(2)) - 5

    # Mask used to extract n LSBs of an integer
    ctr_mask = (1 << interleaver_param) - 1

    # Generate interleaver and deinterleaver
    turbo_int = [-1 for n in range(n_bits)]
    turbo_deint = [-1 for n in range(n_bits)]
    in_addr = 0
    for ctr in range(1 << (interleaver_param + 5)):
        msb, lsb = ctr >> 5, ctr & 31
        new_val = (msb + 1) & ctr_mask
        lut_out = int_table[interleaver_param - 3][lsb]
        out_addr = ((bit_rev32[lsb] << interleaver_param)
                    + ((new_val * lut_out) & ctr_mask))
        if out_addr < n_bits:
            turbo_int[in_addr], turbo_deint[out_addr] = out_addr, in_addr
            in_addr += 1

    return turbo_int, turbo_deint


class Conv(object):
    '''
    Encoder and decoder for a binary convolutional code of rate 1/N
    (1 input bit stream and N output bit streams).

    Attributes
    ----------
    mem_len : int
        Number of bits in encoder state.
    state_space : tuple
        Encoder state space, taken as all integers in [0, 2 ** mem_len).
    n_out : int
        Number of encoder output bits per input bit.
    next_state_msb : tuple
        Tuple of length 2, with next_state_msb[b] being a tuple of length
        2 ** mem_len, and next_state_msb[b][s] being the MSB of the next
        state when the current state is s and the input bit is b.
    out_bits : tuple
        Tuple of length 2, with out_bits[b] being a tuple of length
        2 ** mem_len, and out_bits[b][s] being a tuple of length n_out
        representing the output bits when the current state is s and
        the input bit is b.
    next_state : tuple
        Tuple of length 2, with next_state[b] being a tuple of length
        2 ** mem_len, and next_state[b][s] being the next state when the
        current state is s and the input bit is b.
    '''

    INF = 1e6

    def __init__(self, back_poly, fwd_polys):
        '''
        Init method.

        Parameters
        ----------
        back_poly : int
            For a code of constraint length L, back_poly must be in the
            range [2^(L-1), 2^L).
        fwd_polys : tuple
            For a code of constraint length L and rate 1/N, fwd_polys must
            be of length N, with int elements in the range [1, 2^L).

        Notes
        -----
        The generator polynomials for the binary convolutional code are
        specified through back_poly (positive integer) and fwd_polys
        (tuple of positive integers). For a code of constraint length L
        and rate 1/N:
        (a) back_poly must be in the range [2^(L-1), 2^L);
        (b) fwd_polys must be of length N and its elements must be
        in the range [1, 2^L).

        Let b[0], b[1],..., b[L-1] be the binary representation
        of back_poly, with b[0] = 1 being the MSB. Similarly, let
        f[n][0], f[n][1],..., f[n][L-1] be the L-bit binary
        representation of fwd_polys[n], with f[n][0] being the MSB.

        The input-output relationship of the encoder can then be
        described as follows. Let x[k] be the input bit at time k
        to the encoder, and let y[n][k], n = 0,1,...,N-1, be the
        n^th output bit at time k from the encoder. Then,
        y[n][k] = sum_{i=0}^{L-1} f[n][i] * s[k-i],
        where the sequence of bits s[k] is given by
        s[k] = b[0] * x[k] + sum_{i=1}^{L-1} b[i] * s[k-i].
        The encoder has 2^(L-1) states, with the state at time k
        being comprised of the bits s[k-1], s[k-2],..., s[k-L+1].

        The output bits at time k are read out in the order
        y[0][k], y[1][k],..., y[N-1][k].

        For the constituent recursive systematic convolutional (RSC)
        code in the 3GPP/3GPP2 turbo code, set back_poly = 11, and
        fwd_polys = (11, 13, 15).
        '''

        # Number of bits in encoder state.
        self.mem_len = math.floor(math.log(back_poly) / math.log(2))

        # Encoder state space (integers in the range [0, 2 ** mem_len)).
        self.state_space = tuple(n for n in range(1 << self.mem_len))

        # Number of encoder output bits per input bit.
        self.n_out = len(fwd_polys)

        # MSB of next encoder state, given current state and input bit.
        self.next_state_msb = tuple(tuple(
            bitxor(back_poly & ((b << self.mem_len) + s))
            for s in self.state_space) for b in (0, 1))

        # Encoder output bits, given current state and input bit.
        self.out_bits = tuple(tuple(tuple(
            bitxor(p & ((self.next_state_msb[b][s] << self.mem_len) + s))
            for p in fwd_polys) for s in self.state_space) for b in (0, 1))

        # Next encoder state, given current state and input bit.
        self.next_state = tuple(tuple(
            (self.next_state_msb[b][s] << (self.mem_len - 1)) + (s >> 1)
            for s in self.state_space) for b in (0, 1))

        return

    def encode(self, info_bits):
        '''
        Encodes a given sequence of info bits (does not modify self).

        Parameters
        ----------
        info_bits : ndarray of dtype int
            Array specifying the info bits (0 or 1) to be encoded.

        Returns
        -------
        code_bits : ndarray of dtype int
            1-dim array of size self.n_out * (info_bits.size
            + self.mem_len), giving the resulting code bits.

        Notes
        -----
        The encoder begins in state 0, and is brought back to state 0
        at the end with self.mem_len tail bits.
        '''

        info_bits = np.asarray(info_bits).ravel()
        n_info_bits = info_bits.size

        code_bits, enc_state = -np.ones(
            self.n_out * (n_info_bits + self.mem_len), dtype=int), 0
        for k in range(n_info_bits + self.mem_len):
            in_bit = (info_bits[k] if k < n_info_bits
                      else self.next_state_msb[0][enc_state])
            code_bits[self.n_out * k : self.n_out * (k + 1)] = (
                self.out_bits[in_bit][enc_state])
            enc_state = self.next_state[in_bit][enc_state]

        return code_bits

    def _branch_metrics(self, out_bit_llrs, pre_in_bit_llr=0):
        '''
        Computes branch metrics (does not modify self).

        Parameters
        ----------
        out_bit_llrs : tuple
            Tuple of length self.n_out with float elements, specifying
            the LLRs for the output bits.
        pre_in_bit_llr : float
            Prior LLR for the input bit.

        Returns
        -------
        gamma_val : tuple
            Tuple of length 2, with gamma_val[b] being a list of length
            len(self.state_space) giving the branch metrics for the
            transitions out of all states when the input bit is b.
        '''

        gamma_val = ([pre_in_bit_llr / 2 for s in self.state_space],
                     [-pre_in_bit_llr / 2 for s in self.state_space])
        for enc_state in self.state_space:
            for bit0, bit1, val in zip(self.out_bits[0][enc_state],
                                       self.out_bits[1][enc_state],
                                       out_bit_llrs):
                gamma_val[0][enc_state] += val / 2 if bit0 == 0 else -val / 2
                gamma_val[1][enc_state] += val / 2 if bit1 == 0 else -val / 2

        return gamma_val

    def _update_path_metrics(self, out_bit_llrs, path_metrics, best_bit):
        '''
        Updates path metrics and finds best input bits for all states
        (does not modify self).

        Parameters
        ----------
        out_bit_llrs : tuple
            Tuple of length self.n_out with float elements, specifying
            the LLRs for the output bits at time k.
        path_metrics : list
            List of length len(self.state_space) with float elements,
            specifying the path metric for each state at time k+1. This
            list is modified in place to contain the path metrics of all
            states at time k.
        best_bit : list
            List of length len(self.state_space). This list is modified in
            place to contain the best input bit in each state at time k.
        '''

        gamma_val = self._branch_metrics(out_bit_llrs)

        pmn = path_metrics[:]
        for enc_state in self.state_space:
            cpm0 = gamma_val[0][enc_state] + pmn[self.next_state[0][enc_state]]
            cpm1 = gamma_val[1][enc_state] + pmn[self.next_state[1][enc_state]]
            path_metrics[enc_state], best_bit[enc_state] = (
                (cpm0, 0) if cpm0 >= cpm1 else (cpm1, 1))

        return

    def decode_viterbi(self, code_bit_llrs):
        '''
        Decoder based on Viterbi algorithm (does not modify self).

        Parameters
        ----------
        code_bit_llrs : ndarray of dtype float
            Array of code bit LLRs (+ve for 0, -ve for 1).

        Returns
        -------
        info_bits_hat : ndarray of dtype int
            1-dim array of size code_bit_llrs.size / self.n_out
            - self.mem_len, giving the decoded info bits.

        Notes
        -----
        The encoder is assumed to have begun in state 0 and to have been
        brought back to state 0 at the end with self.mem_len tail bits.
        '''

        code_bit_llrs = np.asarray(code_bit_llrs).ravel()
        n_in_bits = int(code_bit_llrs.size / self.n_out)
        n_info_bits = n_in_bits - self.mem_len

        # Path metric for each state at time n_in_bits.
        path_metrics = [(0 if s == 0 else -Conv.INF) for s in self.state_space]

        # Best input bit in each state at times 0 to n_in_bits - 1.
        best_bit = [[-1 for s in self.state_space] for k in range(n_in_bits)]

        # Start at time n_in_bits - 1 and work backward to time 0, finding
        # path metric and best input bit for each state at each time.
        for k in range(n_in_bits - 1, -1, -1):
            self._update_path_metrics(
                code_bit_llrs[self.n_out * k : self.n_out * (k + 1)],
                path_metrics, best_bit[k])

        # Decode by starting in state 0 at time 0 and tracing path
        # corresponding to best input bits.
        info_bits_hat, enc_state = -np.ones(n_info_bits, dtype=int), 0
        for k in range(n_info_bits):
            info_bits_hat[k] = best_bit[k][enc_state]
            enc_state = self.next_state[info_bits_hat[k]][enc_state]

        return info_bits_hat

    def _update_alpha(
            self,
            out_bit_llrs,
            pre_in_bit_llr,
            alpha_val,
            alpha_val_next,
            max_log):
        '''
        Computes alpha value for each state at time k+1, given output
        bit LLRs for time k and prior input bit LLR for time k, and
        alpha value for each state at time k. Does not modify self.

        Parameters
        ----------
        out_bit_llrs : tuple
            Tuple of length self.n_out with float elements, specifying
            the LLRs for the output bits at time k.
        pre_in_bit_llr : float
            Prior LLR for the input bit at time k.
        alpha_val : list
            List of length len(self.state_space) specifying alpha values
            for all states at time k.
        alpha_val_next : list
            List of length len(self.state_space). This list is modified
            in place to contain alpha values for all states at time k+1.
        max_log : bool
            Set to True to use the max-log approximation.
        '''

        gamma_val = self._branch_metrics(out_bit_llrs, pre_in_bit_llr)

        for enc_state in self.state_space:
            alpha_val_next[self.next_state[0][enc_state]] = maxstar(
                alpha_val_next[self.next_state[0][enc_state]],
                alpha_val[enc_state] + gamma_val[0][enc_state],
                max_log)
            alpha_val_next[self.next_state[1][enc_state]] = maxstar(
                alpha_val_next[self.next_state[1][enc_state]],
                alpha_val[enc_state] + gamma_val[1][enc_state],
                max_log)

        return

    def _update_beta_tail(self, out_bit_llrs, beta_val, max_log):
        '''
        Computes beta value for each state at time k, given output
        bit LLRs for time k and beta value for each state at time k+1.
        The prior input bit LLR for time k is taken as 0. Does not
        modify self.

        Parameters
        ----------
        out_bit_llrs : tuple
            Tuple of length self.n_out with float elements, specifying
            the LLRs for the output bits at time k.
        beta_val : list
            List of length len(self.state_space) specifying beta values
            for all states at time k+1. This list is modified in place
            to contain beta values for all states at time k.
        max_log : bool
            Set to True to use the max-log approximation.
        '''

        gamma_val = self._branch_metrics(out_bit_llrs, 0)

        bvn = beta_val[:]
        for enc_state in self.state_space:
            beta_val[enc_state] = maxstar(
                gamma_val[0][enc_state] + bvn[self.next_state[0][enc_state]],
                gamma_val[1][enc_state] + bvn[self.next_state[1][enc_state]],
                max_log)

        return

    def _update_beta(
            self,
            out_bit_llrs,
            pre_in_bit_llr,
            alpha_val,
            beta_val,
            max_log):
        '''
        Computes beta value for each state at time k, given output
        bit LLRs and prior input bit LLR for time k, and beta value for
        each state at time k+1. Also computes and returns posterior input
        bit LLR for time k, given alpha value for each state at time k.
        Does not modify self.

        Parameters
        ----------
        out_bit_llrs : tuple
            Tuple of length self.n_out with float elements, specifying
            the LLRs for the output bits at time k.
        pre_in_bit_llr : float
            Prior LLR for the input bit at time k.
        alpha_val : list
            List of length len(self.state_space) specifying alpha values
            for all states at time k.
        beta_val : list
            List of length len(self.state_space) specifying beta values
            for all states at time k+1. This list is modified in place
            to contain beta values for all states at time k.
        max_log : bool
            Set to True to use the max-log approximation.

        Returns
        -------
        post_info_bit_llr : float
            Posterior LLR for the input bit at time k.
        '''

        gamma_val = self._branch_metrics(out_bit_llrs, pre_in_bit_llr)

        met0 = -Conv.INF
        met1 = -Conv.INF
        bvn = beta_val[:]
        for enc_state in self.state_space:
            beta_val[enc_state] = maxstar(
                gamma_val[0][enc_state] + bvn[self.next_state[0][enc_state]],
                gamma_val[1][enc_state] + bvn[self.next_state[1][enc_state]],
                max_log)
            met0 = maxstar(
                alpha_val[enc_state] + gamma_val[0][enc_state]
                + bvn[self.next_state[0][enc_state]],
                met0,
                max_log)
            met1 = maxstar(
                alpha_val[enc_state] + gamma_val[1][enc_state]
                + bvn[self.next_state[1][enc_state]],
                met1,
                max_log)

        return met0 - met1

    def decode_bcjr(
            self,
            code_bit_llrs,
            pre_info_bit_llrs=None,
            max_log=False):
        '''
        Decoder based on BCJR algorithm (does not modify self).

        Parameters
        ----------
        code_bit_llrs : ndarray of dtype float
            Array of code bit LLRs (+ve for 0, -ve for 1).
        pre_info_bit_llrs : ndarray of dtype float
            Array of size code_bit_llrs.size / self.n_out - self.mem_len,
            specifying pre-decoding info bit LLRs (+ve for 0, -ve for 1).
            If set to None, then all such LLRs are taken as 0.
        max_log : bool
            Set to True to use the max-log approximation.

        Returns
        -------
        post_info_bit_llrs : ndarray of dtype float
            1-dim array of size code_bit_llrs.size / self.n_out
            - self.mem_len, giving the post-decoding info bit LLRs.

        Notes
        -----
        The encoder is assumed to have begun in state 0 and to have been
        brought back to state 0 at the end with self.mem_len tail bits.

        The computation of the post-decoding info bit LLRs assumes that
        code_bit_llrs was generated by BPSK-modulating the encoder output
        (+sqrt(Es) for 0 and -sqrt(Es) for 1), passing the BPSK symbols
        through a memoryless real AWGN channel with noise of zero mean
        and variance N0 / 2, and scaling the resulting channel output
        by 4 * sqrt(Es) / N0.

        The post-decoding LLR for info bit k is M(0,k) - M(1,k), where
        M(b,k) = maxstar[alpha(s,k) + gamma(s,ss,k) + beta(ss,k+1)],
        with the maxstar being over all state transitions from s at
        time k to ss at time k+1 that correspond to an input bit of b.

        Here, gamma(s,ss,k) is the branch metric for the transition.
        The alpha and beta values are respectively obtained by the
        Viterbi-like "forward" and "backward" recursions
        alpha(s,k+1) = maxstar[alpha(ss,k) + gamma(ss,s,k)] over all
        states ss at time k, and
        beta(s,k) = maxstar[gamma(s,ss,k) + beta(ss,k+1)] over all
        states ss at time k+1. Both recursions are initialized with
        state 0 having metric 0 and all other states having metrics
        of -Inf.
        '''

        code_bit_llrs = np.asarray(code_bit_llrs).ravel()
        n_in_bits = int(code_bit_llrs.size / self.n_out)
        n_info_bits = n_in_bits - self.mem_len

        if pre_info_bit_llrs is None:
            pre_info_bit_llrs = np.zeros(n_info_bits)
        else:
            pre_info_bit_llrs = np.asarray(pre_info_bit_llrs).ravel()

        # FORWARD PASS: Recursively compute alpha values for all states at
        # all times from 1 to n_info_bits - 1, working forward from time 0.
        alpha = [[(0 if s == 0 and k == 0 else -Conv.INF)
                  for s in self.state_space] for k in range(n_info_bits)]
        for k in range(n_info_bits - 1):
            out_bit_llrs = code_bit_llrs[
                self.n_out * k : self.n_out * (k + 1)]
            self._update_alpha(
                out_bit_llrs, pre_info_bit_llrs[k],
                alpha[k], alpha[k + 1], max_log)

        # BACKWARD PASS (TAIL): Recursively compute beta values for all
        # states at time n_info_bits, working backward from time n_in_bits.
        beta = [(0 if s == 0 else -Conv.INF) for s in self.state_space]
        for k in range(n_in_bits - 1, n_info_bits - 1, -1):
            out_bit_llrs = code_bit_llrs[self.n_out * k : self.n_out * (k + 1)]
            self._update_beta_tail(out_bit_llrs, beta, max_log)

        # BACKWARD PASS: Recursively compute beta values for all states at
        # each time k from 0 to n_info_bits - 1, working backward from time
        # n_info_bits, and also obtaining the post-decoding LLR for the info
        # bit at each time.
        post_info_bit_llrs = np.zeros_like(pre_info_bit_llrs)
        for k in range(n_info_bits - 1, - 1, -1):
            out_bit_llrs = code_bit_llrs[self.n_out * k : self.n_out * (k + 1)]
            post_info_bit_llrs[k] = self._update_beta(
                out_bit_llrs, pre_info_bit_llrs[k],
                alpha[k], beta, max_log)

        return post_info_bit_llrs


class Turbo(object):
    '''
    Encoder and decoder for a turbo code, formed by the parallel
    concatenation of two identical binary recursive systematic
    convolutional (RSC) codes of rate 1/N (1 input bit stream and
    N output bit streams). The turbo code is then of rate 1/(2*N-1).
    The turbo interleaver is from 3GPP2.

    Attributes
    ----------
    rsc : Conv
        Encoder and decoder for constituent RSC code.
    n_out : int
        Number of output bits per input bit.
    n_tail_bits : int
        Number of tail bits per input block.
    turbo_int : list
        List with int elements specifying the turbo interleaver.
    turbo_deint : list
        List with int elements specifying the turbo deinterleaver.
    '''

    def __init__(self, back_poly, parity_polys):
        '''
        Init method.

        Parameters
        ----------
        back_poly : int
            For a constituent RSC code of constraint length L, back_poly
            must be in the range [2^(L-1), 2^L).
        parity_polys : tuple
            For a constituent RSC code of constraint length L and rate
            1/N, parity_polys must be of length N-1, with int elements
            in the range [1, 2^L).

        Notes
        -----
        The generator polynomials for the constituent RSC codes are
        specified through back_poly (positive integer) and parity_polys
        (sequence of positive integers). For a constituent RSC code of
        constraint length L and rate 1/N:
        (a) back_poly must be in the range [2^(L-1), 2^L);
        (b) parity_polys must be of length N-1 and its elements
        must be in the range [1, 2^L).
        The rate of the resulting turbo code is 1/(2*N-1).

        Let b[0], b[1],..., b[L-1] be the binary representation of
        back_poly, with b[0] = 1 being the MSB. Similarly, let
        f[n][0], f[n][1],..., f[n][L-1] be the L-bit binary
        representation of parity_polys[n], with f[n][0] being the MSB.

        The input-output relationship of each RSC encoder can then
        be described as follows. Let x[k] be the input bit at time
        k to the encoder, and let y[n][k], n = 0,1,...,N-1, be the
        n^th output bit at time k from the encoder. Then,
        y[0][k] = x[k] (systematic bit) and, for n = 1,2,...,N-1,
        y[n][k] = sum_{i=0}^{L-1} f[n-1][i] * s[k-i],
        where the sequence of bits s[k] is given by
        s[k] = b[0]*x[k] + sum_{i=1}^{L-1} b[i] * s[k-i].
        The encoder has 2^(L-1) states, with the state at time k
        being comprised of the bits s[k-1], s[k-2],..., s[k-L+1].

        The output bits at time k from each constituent encoder are
        read out in the order y[0][k], y[1][k],..., y[N-1][k]. For
        the final turbo code output, the systematic bit from the
        bottom encoder is suppressed. The output bits from the top
        encoder are followed by just the parity outputs from the
        bottom encoder. At the end, all the tail bits from the top
        encoder are read out, followed by all the tail bits from
        the bottom encoder.

        For the 3GPP/3GPP2 turbo code, set back_poly = 11, and
        parity_polys = [13, 15].
        '''

        # Encoder and decoder for constituent RSC code
        self.rsc = Conv(back_poly, [back_poly] + parity_polys)

        # Number of output bits per input bit and number of tail bits
        # per input block for the turbo code
        self.n_out = self.rsc.n_out + (self.rsc.n_out - 1)
        self.n_tail_bits = self.rsc.n_out * self.rsc.mem_len * 2

        # Turbo interleaver and deinterleaver
        self.turbo_int, self.turbo_deint = [], []

        return

    def encode(self, info_bits):
        '''
        Encodes a given sequence of info bits (could modify self).

        Parameters
        ----------
        info_bits : ndarray of dtype int
            Array specifying the info bits (0 or 1) to be encoded.

        Returns
        -------
        code_bits : ndarray of dtype int
            1-dim array of size self.n_out * info_bits.size
            + self.n_tail_bits, giving the resulting code bits.

        Notes
        -----
        Both top and bottom encoders begin in state 0, and are brought
        back to state 0 at the end with self.rsc.mem_len tail bits.
        '''

        info_bits = np.asarray(info_bits).ravel()
        n_info_bits = info_bits.size

        if n_info_bits != len(self.turbo_int):
            self.turbo_int, self.turbo_deint = turbo_int_3gpp2(n_info_bits)

        # Get code bits from each encoder.
        ctop = self.rsc.encode(info_bits)
        cbot = self.rsc.encode(info_bits[self.turbo_int])

        # Assemble code bits from both encoders.
        code_bits, pos = -np.ones(
            self.n_out * n_info_bits + self.n_tail_bits, dtype=int), 0
        for k in range(n_info_bits):
            code_bits[pos : pos + self.rsc.n_out] = ctop[
                self.rsc.n_out * k : self.rsc.n_out * (k + 1)]
            pos += self.rsc.n_out
            code_bits[pos : pos + self.rsc.n_out - 1] = cbot[
                self.rsc.n_out * k + 1 : self.rsc.n_out * (k + 1)]
            pos += self.rsc.n_out - 1
        code_bits[pos : pos + self.rsc.n_out * self.rsc.mem_len] = ctop[
            self.rsc.n_out * n_info_bits :]
        code_bits[pos + self.rsc.n_out * self.rsc.mem_len :] = cbot[
            self.rsc.n_out * n_info_bits :]

        return code_bits

    def decode(self, code_bit_llrs, n_turbo_iters, max_log=False):
        '''
        Decoder based on specified number of turbo iterations (could modify
        self).

        Parameters
        ----------
        code_bit_llrs : ndarray of dtype float
            Array of code bit LLRs (+ve for 0, -ve for 1).
        n_turbo_iters : int
            Number of turbo iterations to run.
        max_log : bool
            Set to True to use the max-log approximation.

        Returns
        -------
        info_bits_hat : ndarray of dtype int
            1-dim array of size (code_bit_llrs.size - self.n_tail_bits)
            / self.n_out, giving the decoded info bits.
        post_info_bit_llrs : ndarray of dtype float
            1-dim array of size (code_bit_llrs.size - self.n_tail_bits)
            / self.n_out, giving the post-decoding info bit LLRs.

        Notes
        -----
        Both top and bottom encoders are assumed to have begun in state 0
        and to have been brought back to state 0 at the end with
        self.rsc.mem_len tail bits.
        '''

        code_bit_llrs = np.asarray(code_bit_llrs).ravel()
        n_info_bits = int((code_bit_llrs.size - self.n_tail_bits) / self.n_out)

        if n_info_bits != len(self.turbo_int):
            self.turbo_int, self.turbo_deint = turbo_int_3gpp2(n_info_bits)

        # Systematic bit LLRs for each decoder
        sys_llrs_top = code_bit_llrs[0 : self.n_out * n_info_bits : self.n_out]
        sys_llrs_bot = sys_llrs_top[self.turbo_int]

        # Code bit LLRs for each decoder
        ctop_llrs = np.zeros(self.rsc.n_out * (n_info_bits + self.rsc.mem_len))
        cbot_llrs = np.zeros(self.rsc.n_out * (n_info_bits + self.rsc.mem_len))
        pos = 0
        for k in range(n_info_bits):
            num = self.rsc.n_out * k
            ctop_llrs[num] = sys_llrs_top[k]
            cbot_llrs[num] = sys_llrs_bot[k]
            pos += 1
            ctop_llrs[num + 1 : num + self.rsc.n_out] = code_bit_llrs[
                pos : pos + self.rsc.n_out - 1]
            pos += self.rsc.n_out - 1
            cbot_llrs[num + 1 : num + self.rsc.n_out] = code_bit_llrs[
                pos : pos + self.rsc.n_out - 1]
            pos += self.rsc.n_out - 1
        ctop_llrs[self.rsc.n_out * n_info_bits :] = code_bit_llrs[
            pos : pos + self.rsc.n_out * self.rsc.mem_len]
        cbot_llrs[self.rsc.n_out * n_info_bits :] = code_bit_llrs[
            pos + self.rsc.n_out * self.rsc.mem_len :]

        # Main loop for turbo iterations
        ipre_llrs, ipost_llrs = np.zeros(n_info_bits), np.zeros(n_info_bits)
        for _ in range(n_turbo_iters):
            ipost_llrs[:] = self.rsc.decode_bcjr(ctop_llrs, ipre_llrs, max_log)
            ipre_llrs[:] = (ipost_llrs[self.turbo_int]
                            - ipre_llrs[self.turbo_int]
                            - sys_llrs_top[self.turbo_int])
            ipost_llrs[:] = self.rsc.decode_bcjr(cbot_llrs, ipre_llrs, max_log)
            ipre_llrs[:] = (ipost_llrs[self.turbo_deint]
                            - ipre_llrs[self.turbo_deint]
                            - sys_llrs_bot[self.turbo_deint])

        # Final post-decoding LLRs and hard decisions
        post_info_bit_llrs = ipost_llrs[self.turbo_deint]
        info_bits_hat = (post_info_bit_llrs < 0).astype(int)

        return info_bits_hat, post_info_bit_llrs
