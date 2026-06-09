import streamlit as st
import matplotlib.pyplot as plt

from train import TrainAgent

# ====================================
# PAGE CONFIG
# ====================================

st.set_page_config(
    page_title="Autonomous Paddle Control",
    layout="wide"
)

# ====================================
# SIDEBAR
# ====================================

with st.sidebar:

    st.header("Training Configuration")

    Episodes = st.slider(
        "Training Episodes",
        min_value=100,
        max_value=10000,
        value=3000,
        step=100
    )

    st.divider()

    StartTraining = st.button(
        "▶ Start Training",
        use_container_width=True
    )

# ====================================
# TITLE
# ====================================

st.title(
    "🎮 Autonomous Paddle Control Using DDQN"
)

st.markdown(
    """
    Deep Reinforcement Learning based paddle-ball game.
    The agent learns to intercept the ball and maximize reward.
    """
)

# ====================================
# LIVE GAME AREA
# ====================================

st.subheader("🎯 Live Simulation")

GameArea = st.empty()

GameArea.info(
    "Live game visualization will appear here."
)

# ====================================
# TRAINING PROGRESS
# ====================================

st.subheader("🚀 Training Progress")

ProgressText = st.empty()

ProgressBar = st.progress(0)

MetricCol1, MetricCol2, MetricCol3 = st.columns(3)

CurrentRewardMetric = MetricCol1.empty()
BestRewardMetric = MetricCol2.empty()
AverageRewardMetric = MetricCol3.empty()

CurrentRewardMetric.metric(
    "Current Reward",
    0
)

BestRewardMetric.metric(
    "Best Reward",
    0
)

AverageRewardMetric.metric(
    "Average Reward",
    0
)

# ====================================
# GRAPH
# ====================================

st.subheader("📈 Reward Progress Graph")

GraphPlaceholder = st.empty()

# ====================================
# RESULTS
# ====================================

st.subheader("🏆 Training Result")

ResultPlaceholder = st.empty()

# ====================================
# TRAINING
# ====================================

if StartTraining:

    FinalRewards = []

    for Result in TrainAgent(Episodes):

        CurrentEpisode = Result["Episode"]

        CurrentReward = Result["Reward"]

        BestReward = Result["BestReward"]

        AverageReward = Result["AverageReward"]

        Rewards = Result["Rewards"]

        FinalRewards = Rewards

        ProgressBar.progress(
            CurrentEpisode / Episodes
        )

        ProgressText.info(
            f"Episode {CurrentEpisode} / {Episodes}"
        )

        CurrentRewardMetric.metric(
            "Current Reward",
            CurrentReward
        )

        BestRewardMetric.metric(
            "Best Reward",
            BestReward
        )

        AverageRewardMetric.metric(
            "Average Reward",
            round(AverageReward, 2)
        )

        if CurrentEpisode % 10 == 0:

            Figure, Axis = plt.subplots(
                figsize=(10, 4)
            )

            Axis.plot(
                Rewards,
                linewidth=2
            )

            Axis.set_title(
                "Reward Progress"
            )

            Axis.set_xlabel(
                "Episode"
            )

            Axis.set_ylabel(
                "Reward"
            )

            Axis.grid(True)

            GraphPlaceholder.pyplot(
                Figure
            )

            plt.close()

    # =================================
    # FINAL RESULT
    # =================================

    PositiveEpisodes = sum(
        1
        for Reward in FinalRewards
        if Reward > 0
    )

    SuccessRate = (
        PositiveEpisodes /
        len(FinalRewards)
    ) * 100

    ResultPlaceholder.success(
        f"""
Training Completed Successfully

Episodes Trained : {Episodes}

Best Reward : {max(FinalRewards)}

Average Reward : {sum(FinalRewards)/len(FinalRewards):.2f}

Successful Episodes : {PositiveEpisodes}

Success Rate : {SuccessRate:.2f}%
"""
    )

    st.balloons()