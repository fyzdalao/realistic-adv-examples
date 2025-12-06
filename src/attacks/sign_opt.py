"""
SignOPT 攻击实现

【重要问题说明：为什么 SignOPT 在强防御下会直接失败？】

SignOPT 有一个严格的提前终止机制，导致它在强防御下连让模型分类错误都做不到：

1. 【初始方向搜索的严格条件】（第74-91行）
   - 只尝试 num_directions 次（默认100次）随机方向
   - 要求 x + theta 必须直接能让模型分类错误（success == True）
   - 如果所有尝试都失败，g_theta 保持为 float("inf")

2. 【提前终止机制】（第93-95行）
   - 如果 g_theta == inf，直接返回失败
   - 返回原始图像 x，距离为 inf，成功标志为 False
   - 根本没有进入后续的梯度下降优化阶段

3. 【与其他攻击的对比】
   - HSJA: 持续尝试随机噪声直到找到对抗样本（最多1e4次），即使初始距离很大也会继续优化
   - OPT 基类: 有两次尝试机会，而 SignOPT 只尝试一次
   - RayS/GeoDA: 即使初始距离很大，也会继续优化，最终可能找到对抗样本

4. 【在强防御下的表现】
   - 随机方向 x + theta 很难直接让模型分类错误
   - 所有 num_directions 次尝试都失败
   - 攻击在初始阶段就放弃，返回失败
   - 其他攻击虽然最终距离很大，但至少能让模型分类错误

【关键代码位置】
- 第78-91行：初始方向搜索循环
- 第80行：检查 x + theta 是否能让模型分类错误
- 第93-95行：提前终止机制（直接返回失败）
- 第104-196行：梯度下降优化阶段（如果初始搜索失败，永远不会执行到这里）
"""

import itertools
import warnings
from typing import Tuple

import numpy as np
import torch
from foolbox.distances import LpDistance

from src.attacks.base import Bounds, ExtraResultsDict, SearchMode
from src.attacks.opt import OVERSHOOT_VALUE, OPT, OPTAttackPhase, normalize
from src.attacks.queries_counter import QueriesCounter
from src.model_wrappers import ModelWrapper

start_learning_rate = 1.0


class SignOPT(OPT):

    def __init__(
        self,
        epsilon: float | None,
        distance: LpDistance,
        bounds: Bounds,
        discrete: bool,
        queries_limit: int | None,
        unsafe_queries_limit: int | None,
        max_iter: int | None,
        alpha: float,
        beta: float,
        num_grad_queries: int,
        search: SearchMode,
        grad_estimation_search: SearchMode,
        step_size_search: SearchMode,
        n_searches: int,
        max_search_steps: int,
        momentum: float = 0.,
        batch_size: int | None = None,
        num_init_directions: int = 100,
        get_one_init_direction: bool = False,
    ):
        super().__init__(epsilon, distance, bounds, discrete, queries_limit, unsafe_queries_limit, max_iter, alpha,
                         beta, search, num_grad_queries, grad_estimation_search, step_size_search, n_searches,
                         max_search_steps, batch_size, num_init_directions, get_one_init_direction)
        self.momentum = momentum  # (default: 0)
        if batch_size is not None:
            self.grad_batch_size = min(batch_size, self.num_grad_queries)
        else:
            self.grad_batch_size = self.num_grad_queries

        # Args needed for targeted attack
        # self.tgt_init_query = args["signopt_tgt_init_query"]
        # self.targeted_dataloader = targeted_dataloader

    def __call__(
            self,
            model: ModelWrapper,
            x: torch.Tensor,
            label: torch.Tensor,
            target: torch.Tensor | None = None) -> tuple[torch.Tensor, QueriesCounter, float, bool, ExtraResultsDict]:
        if target is not None:
            if self.momentum > 0:
                warnings.warn("Currently, targeted Sign-OPT does not support momentum, ignoring argument.")
            raise NotImplementedError('Targeted attack is not implemented for OPT')
        return self.attack_untargeted(model, x, label)

    def attack_untargeted(self, model: ModelWrapper, x: torch.Tensor,
                          y: torch.Tensor) -> tuple[torch.Tensor, QueriesCounter, float, bool, ExtraResultsDict]:
        """Attack the original image and return adversarial example
        (x0, y0): original image
        """
        queries_counter = self._make_queries_counter()
        target = None

        # ========================================================================
        # 【关键问题所在】初始方向搜索阶段 - 这是导致攻击直接失败的根本原因
        # ========================================================================
        # SignOPT 要求必须找到一个能让模型分类错误的初始方向，否则直接返回失败。
        # 这与 HSJA 等其他攻击不同，它们即使初始距离很大也会继续优化。
        # ========================================================================
        # Calculate a good starting point.
        best_theta, g_theta = None, float("inf")
        if self.verbose:
            print(f"Searching for the initial direction on {self.num_directions} random directions: ")
        
        # 【问题1】只尝试 num_directions 次（默认100次）随机方向
        # 如果这100次都找不到能让模型分类错误的方向，g_theta 会保持为 inf
        for i in range(self.num_directions):
            theta = torch.randn_like(x)
            # 【问题2】关键检查：要求 x + theta 必须直接能让模型分类错误
            # is_correct_boundary_side 的实现（base.py 第35-51行）：
            #   - 对于非目标攻击（target is None），检查 model.predict_label(x_adv) != y
            #   - 如果模型在 x + theta 上的预测 != 原始标签 y，返回 success = True
            #   - success = True 表示模型分类错误（即 x + theta 是"安全"的，在边界错误一侧）
            # 
            # 【在强防御下的问题】：
            # - 随机方向 x + theta 很难直接让模型分类错误
            # - 导致 success 总是 False，所有方向都被跳过
            # - 这个循环结束后，g_theta 仍然是 inf，没有任何方向被处理
            success, queries_counter = self.is_correct_boundary_side(model, x + theta, y, target, queries_counter,
                                                                     OPTAttackPhase.direction_search, x)
            # 【问题3】只有 success == True 时才会继续处理这个方向
            # - 如果 success == True：归一化方向，进行精细搜索，更新 best_theta 和 g_theta
            # - 如果 success == False：直接跳过这个方向，不进行任何处理
            # - 如果所有方向都让 success == False，这个循环结束后 g_theta 仍然是 inf
            if success.item():
                theta, initial_lbd = normalize(theta)
                lbd, queries_counter, _ = self.fine_grained_search(model, x, y, target, theta, queries_counter,
                                                                   initial_lbd.item(), g_theta)
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    if self.verbose:
                        print("--------> Found distortion %.4f" % g_theta)
                if self.get_one_init_direction:
                    break

        # ========================================================================
        # 【提前终止机制】这是导致攻击直接失败的关键代码
        # ========================================================================
        # 如果 g_theta == inf，说明在所有 num_directions 次尝试中，
        # 没有任何一个随机方向能让模型分类错误。
        # 
        # 【对比其他攻击】：
        # - HSJA: 会持续尝试随机噪声直到找到对抗样本（最多1e4次），即使初始距离很大也会继续优化
        # - OPT 基类: 有两次尝试机会（第126-138行和第145-155行），而 SignOPT 只尝试一次
        # - RayS/GeoDA: 即使初始距离很大，也会继续优化，最终可能找到对抗样本
        #
        # 【在强防御下的表现】：
        # 1. 随机方向 x + theta 很难直接让模型分类错误
        # 2. 所有 num_directions 次尝试都失败，g_theta 保持为 inf
        # 3. 攻击在这里直接返回失败，返回原始图像 x，距离为 inf，成功标志为 False
        # 4. 根本没有进入后续的梯度下降优化阶段（第104-196行）
        #
        # 【为什么其他攻击至少能让模型分类错误】：
        # - 它们不会在初始阶段就放弃，即使初始距离很大也会继续优化
        # - 它们可能使用不同的初始化策略，或者有更多的尝试机会
        # ========================================================================
        if g_theta == float("inf"):
            print("Failed to find a good initial direction.")
            # 【直接返回失败】返回原始图像，距离为 inf，成功标志为 False
            # 这意味着攻击完全没有尝试优化，连让模型分类错误都做不到
            return x, queries_counter, float("inf"), False, {}
        else:
            assert best_theta is not None

        if self.verbose:
            print("==========> Found best distortion %.4f "
                  "using %d queries and %d unsafe queries" %
                  (g_theta, queries_counter.total_queries, queries_counter.total_unsafe_queries))

        # ========================================================================
        # 【梯度下降优化阶段】只有成功找到初始方向才会执行到这里
        # ========================================================================
        # 如果代码执行到这里，说明已经找到了一个能让模型分类错误的初始方向。
        # 这个阶段会通过梯度下降来优化对抗样本，减小距离。
        # 
        # 【注意】在强防御下，如果初始方向搜索失败（第93-95行），
        # 代码永远不会执行到这里，攻击会直接返回失败。
        # ========================================================================
        # Begin Gradient Descent.
        xg, gg = best_theta, g_theta
        best_pert = gg * xg
        vg = torch.zeros_like(xg)
        alpha, beta = self.alpha, self.beta
        search_lower_bound = 1 - (OVERSHOOT_VALUE - 1)

        if self.iterations is not None:
            _range = range(self.iterations)
        else:
            _range = itertools.count()

        # 【梯度下降主循环】通过 sign_grad_v2 估计梯度方向，然后进行线搜索优化
        for i in _range:
            sign_gradient, queries_counter = self.sign_grad_v2(model,
                                                               x.squeeze(0),
                                                               y,
                                                               None,
                                                               xg.squeeze(0),
                                                               initial_lbd=gg,
                                                               queries_counter=queries_counter,
                                                               h=beta)

            # Line search
            min_theta = xg
            min_g2 = gg
            min_vg = vg
            for _ in range(15):
                if self.momentum > 0:
                    new_vg = self.momentum * vg - alpha * sign_gradient
                    new_theta = xg + new_vg
                else:
                    new_theta = xg - alpha * sign_gradient
                new_theta, _ = normalize(new_theta)
                new_g2, queries_counter, _ = self.step_size_search_search_fn(model, x, y, target, new_theta,
                                                                             queries_counter, min_g2, beta / 500,
                                                                             search_lower_bound)
                alpha *= 2
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    if self.momentum > 0:
                        min_vg = new_vg  # type: ignore
                else:
                    break

            if min_g2 >= gg:
                for _ in range(15):
                    alpha *= 0.25
                    if self.momentum > 0:
                        new_vg = self.momentum * vg - alpha * sign_gradient
                        new_theta = xg + new_vg
                    else:
                        new_theta = xg - alpha * sign_gradient
                    new_theta, _ = normalize(new_theta)
                    new_g2, queries_counter, _ = self.step_size_search_search_fn(model, x, y, target, new_theta,
                                                                                 queries_counter, min_g2, beta / 500,
                                                                                 search_lower_bound)
                    if new_g2 < gg:
                        min_theta = new_theta
                        min_g2 = new_g2
                        if self.momentum > 0:
                            min_vg = new_vg  # type: ignore
                        break

            if alpha < 1e-4:
                alpha = 1.0
                if self.verbose:
                    print("Warning: not moving")
                beta *= 0.1
                if beta < 1e-8:
                    break

            xg, gg = min_theta, min_g2
            vg = min_vg

            # EDIT: terminate as soon as max queries are used
            if queries_counter.is_out_of_queries():
                break
            best_pert = gg * xg

            if i % 5 == 0 and self.verbose:
                print("Iteration %3d distortion %.4f num_queries %d unsafe queries %d" %
                      (i + 1, gg, queries_counter.total_queries, queries_counter.total_unsafe_queries))

        if self.verbose:
            target = model.predict_label(x + best_pert)
            print("\nAdversarial Example Found Successfully: distortion %.4f target"
                  " %d queries %d unsafe queries %d" %
                  (gg, target, queries_counter.total_queries, queries_counter.total_unsafe_queries))

        x_adv = self.get_x_adv(x, xg, gg)

        return x_adv, queries_counter, gg, True, {}

    def sign_grad_v2(self,
                     model,
                     x: torch.Tensor,
                     y: torch.Tensor,
                     target: torch.Tensor | None,
                     theta: torch.Tensor,
                     initial_lbd: float,
                     queries_counter: QueriesCounter,
                     h: float = 0.001) -> Tuple[torch.Tensor, QueriesCounter]:
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \\sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        
        【注意】这个方法只在梯度下降阶段被调用（第117行）。
        如果初始方向搜索失败，这个方法永远不会被调用。
        """
        sign_grad = torch.zeros_like(theta)
        num_batches = int(np.ceil(self.num_grad_queries / self.grad_batch_size))
        assert num_batches * self.grad_batch_size == self.num_grad_queries
        x = x.unsqueeze(0)
        x_temp = self.get_x_adv(x, theta, initial_lbd)

        for _ in range(num_batches):
            u = torch.randn((self.grad_batch_size, ) + theta.shape, dtype=theta.dtype, device=x.device)
            u, _ = normalize(u, batch=True)

            sign_v = torch.ones((self.grad_batch_size, 1, 1, 1), device=x.device)
            new_theta: torch.Tensor = theta + h * u  # type: ignore
            new_theta, _ = normalize(new_theta, batch=True)

            x_ = self.get_x_adv(x, new_theta, initial_lbd)
            u = x_ - x_temp
            # 【梯度估计中的边界检查】
            # success == True 表示模型在 x_ 上分类错误（即 x_ 是"安全"的，在边界错误一侧）
            # success == False 表示模型在 x_ 上分类正确（即 x_ 是"不安全"的，在边界正确一侧）
            # 这个信息用于估计梯度的符号方向
            success, queries_counter = self.is_correct_boundary_side(model, x_, y, target, queries_counter,
                                                                     OPTAttackPhase.gradient_estimation, x)

            sign_v[success] = -1

            sign_grad += (u.sign() * sign_v).sum(0)

        sign_grad /= self.num_grad_queries
        return sign_grad, queries_counter


def sign(y):
    """
    y -- numpy array of shape (m,)
    Returns an element-wise indication of the sign of a number.
    The sign function returns -1 if y < 0, 1 if x >= 0. nan is returned for nan inputs.
    """
    y_sign = torch.sign(y)
    y_sign[y_sign == 0] = 1
    return y_sign
