import torch
import pandas as pd

from environment import PaddleEnvironment
from ddqn_agent import DDQNAgent


def TrainAgent(Episodes=3000):

    Env = PaddleEnvironment()
    Agent = DDQNAgent()

    Rewards = []

    BestReward = float("-inf")

    for Episode in range(Episodes):

        State = Env.Reset()

        TotalReward = 0

        for _ in range(1000):

            Action = Agent.Act(State)

            NextState, Reward, Done = Env.Step(Action)

            Agent.Remember(
                State,
                Action,
                Reward,
                NextState,
                Done
            )

            Agent.Replay()

            State = NextState

            TotalReward += Reward

            if Done:
                break

        if Episode % 20 == 0:
            Agent.UpdateTarget()

        Rewards.append(TotalReward)

        BestReward = max(
            BestReward,
            TotalReward
        )

        AverageReward = (
            sum(Rewards) / len(Rewards)
        )

        yield {
            "Episode": Episode + 1,
            "Reward": TotalReward,
            "BestReward": BestReward,
            "AverageReward": AverageReward,
            "Rewards": Rewards.copy(),
            "Agent": Agent
        }

    torch.save(
        Agent.PolicyNet.state_dict(),
        "models/ddqn_model.pth"
    )

    pd.DataFrame(
        {"Reward": Rewards}
    ).to_csv(
        "rewards/reward_history.csv",
        index=False
    )