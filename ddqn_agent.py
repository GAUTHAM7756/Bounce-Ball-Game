import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):

    def __init__(self, StateSize, ActionSize):
        super().__init__()

        self.Model = nn.Sequential(
            nn.Linear(StateSize, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, ActionSize)
        )

    def forward(self, X):
        return self.Model(X)


class DDQNAgent:

    def __init__(self):

        self.StateSize = 5
        self.ActionSize = 3

        self.Gamma = 0.99

        self.Epsilon = 1.0
        self.EpsilonMin = 0.01
        self.EpsilonDecay = 0.995

        self.BatchSize = 64

        self.Memory = deque(maxlen=50000)

        self.Device = torch.device("cpu")

        self.PolicyNet = QNetwork(
            self.StateSize,
            self.ActionSize
        ).to(self.Device)

        self.TargetNet = QNetwork(
            self.StateSize,
            self.ActionSize
        ).to(self.Device)

        self.TargetNet.load_state_dict(
            self.PolicyNet.state_dict()
        )

        self.Optimizer = optim.Adam(
            self.PolicyNet.parameters(),
            lr=0.001
        )

    def Remember(
            self,
            State,
            Action,
            Reward,
            NextState,
            Done):

        self.Memory.append(
            (
                State,
                Action,
                Reward,
                NextState,
                Done
            )
        )

    def Act(self, State):

        if random.random() < self.Epsilon:
            return random.randint(0, 2)

        StateTensor = torch.FloatTensor(
            State
        ).unsqueeze(0)

        with torch.no_grad():
            QValues = self.PolicyNet(StateTensor)

        return torch.argmax(QValues).item()

    def Replay(self):

        if len(self.Memory) < self.BatchSize:
            return

        Batch = random.sample(
            self.Memory,
            self.BatchSize
        )

        for State, Action, Reward, NextState, Done in Batch:

            StateTensor = torch.FloatTensor(
                State
            ).unsqueeze(0)

            NextTensor = torch.FloatTensor(
                NextState
            ).unsqueeze(0)

            Target = Reward

            if not Done:

                BestAction = torch.argmax(
                    self.PolicyNet(NextTensor)
                ).item()

                Target += self.Gamma * \
                          self.TargetNet(
                              NextTensor
                          )[0][BestAction].item()

            CurrentQ = self.PolicyNet(
                StateTensor
            )[0][Action]

            Loss = nn.MSELoss()(
                CurrentQ,
                torch.tensor(Target)
            )

            self.Optimizer.zero_grad()
            Loss.backward()
            self.Optimizer.step()

        if self.Epsilon > self.EpsilonMin:
            self.Epsilon *= self.EpsilonDecay

    def UpdateTarget(self):
        self.TargetNet.load_state_dict(
            self.PolicyNet.state_dict()
        )