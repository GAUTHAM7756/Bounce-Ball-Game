import numpy as np

class PaddleEnvironment:

    def __init__(self):
        self.Width = 800
        self.Height = 600

        self.PaddleWidth = 100
        self.PaddleHeight = 15
        self.PaddleSpeed = 15

        self.BallRadius = 10

        self.Reset()

    def Reset(self):

        self.BallX = self.Width // 2
        self.BallY = self.Height // 2

        self.BallVx = np.random.choice([-5, 5])
        self.BallVy = 5

        self.PaddleX = self.Width // 2

        return self.GetState()

    def GetState(self):

        return np.array([
            self.BallX / self.Width,
            self.BallY / self.Height,
            self.BallVx / 10,
            self.BallVy / 10,
            self.PaddleX / self.Width
        ], dtype=np.float32)

    def Step(self, Action):

        Reward = 0
        Done = False

        if Action == 0:
            self.PaddleX -= self.PaddleSpeed

        elif Action == 2:
            self.PaddleX += self.PaddleSpeed

        self.PaddleX = np.clip(
            self.PaddleX,
            self.PaddleWidth // 2,
            self.Width - self.PaddleWidth // 2
        )

        self.BallX += self.BallVx
        self.BallY += self.BallVy

        if self.BallX <= self.BallRadius:
            self.BallVx *= -1

        if self.BallX >= self.Width - self.BallRadius:
            self.BallVx *= -1

        if self.BallY <= self.BallRadius:
            self.BallVy *= -1

        PaddleTop = self.Height - 40

        if self.BallY >= PaddleTop:

            if abs(self.BallX - self.PaddleX) <= self.PaddleWidth // 2:

                Reward = 1
                self.BallVy *= -1

            else:
                Reward = -1
                Done = True

        return self.GetState(), Reward, Done