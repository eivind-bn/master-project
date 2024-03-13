

# def mutate(self, volatility: float, rate: float|None) -> None:
#     for params in self.network.parameters():
#         with torch.no_grad():
#             if rate is None:
#                 params[:] = torch.normal(params, volatility)
#             else:
#                 rands = torch.rand(params.shape, device=params.device)
#                 params[:] = torch.where(rands < rate, torch.normal(params, volatility), params)

# @staticmethod
# def crossover(policies: Sequence["Policy"], weights: Sequence[int|float]) -> "Policy":
#     assert len(policies) == len(weights) and policies
#     total = sum(weights)
#     weight_portions = Stream(weights).map(lambda w: w/total).tuple()

#     result = policies[0].copy()

#     with torch.no_grad():
#         for params in result:
#             params[:] = 0.0

#         for policy,weight_portion in zip(policies,weight_portions, strict=True):
#             for result_param, policy_param in zip(result, policy):
#                 result_param[:] = policy_param*weight_portion

#     return result