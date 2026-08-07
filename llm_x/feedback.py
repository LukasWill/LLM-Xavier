class Feedback:
    def __init__(
        self,
        quest=None,
    ) -> None:
        self.quest = quest
        self.idata=None
        self._data= None

    @property
    def data(self):
        self._data = {
            **self.idata,
            "quest": None
        }
        return self
    def render(self):
        return self._data

##### NextActionPrediction

class NextActionPrediction:

    question_name = "next_action_prediction"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In next step {i} (indexed from 0), the agent transitted to the state s{i} = {state}. Based on your observation and understanding of the agent's behaviour, can you predict the action a{i} (an integer from the given range) the RL agent will most likely take at step {i}? Please first provide a compact reasoning before your answer to the action choice. Think step by step and use the following template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. >>Final action choice: []

        Please choose only one action, even if multiple actions seem possible.
        """
        del task_name

    def render(self, i: int, state):
        return self.question.format(i=i, state=state)

class NextActionPredictionContinuous:  # remove

    question_name = "next_action_prediction_continuous"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In next step {i} (indexed from 0), the agent transitted to the state s{i} = {state}. Based on your observation and understanding of the agent's behaviour, predict the action a{i} that the RL agent will most likely take at state s{i}. You only need to deduce whether each element of the action a{i} increases (symbolized as "INC") or decreases (symbolized as "DEC") compared to the last taken action a{j} = {last_action} at step {j}. Begin with a compact reasoning, followed by a step-by-step prediction for each element of a{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            Action element [i] INC (increase) or DEC (decrease), where "i" indicates the index of the action element (indexed from 0).
            Return a list with the following example format,
            ```python
            # assume action dim is one where element [0] increases
            predictions = ["INC"]
            ```
            Please choose either INC or DEC, even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, j: int, i: int, last_action, state):
        return self.question.format(j=j, i=i, last_action=last_action, state=state)

class NextActionPredictionContinuousMoreOptions:  # remove

    question_name = "next_action_prediction_continuous_more_options"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In next step {i} (indexed from 0), the agent transitted to the state s{i} = {state}. Based on your observation and understanding of the agent's behaviour, predict the action a{i} that the RL agent will most likely take at state s{i}. You only need to deduce whether each element of the action a{i} increases (symbolized as "INC"), decreases (symbolized as "DEC"), or stay unchanged (symbolized as "UNCH"), compared to the last taken action a{j} = {last_action} at step {j}. The action element can stay unchanged if the difference between elements of two successive actions is less than a threshold of 1e-1. Begin with a compact reasoning, followed by a step-by-step prediction for each element of a{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            Action element [i] INC (increase), DEC (decrease), or UNCH (unchange), where "i" indicates the index of the action element (indexed from 0).
            Return a list with the following example format,
            ```python
            # element [0] increases, element [1] decreases, and element [2] remains unchanged
            predictions = ["INC", "DEC", "UNCH"]
            ```
            Note that the provided example needs to be adapted to the current action dim, which is {action_dim}. Please choose either INC, DEC, or UNCH, even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, j: int, i: int, action_dim, last_action, state):
        return self.question.format(j=j, i=i, action_dim=action_dim, last_action=last_action, state=state)

class NextActionPredictionContinuousBins:

    question_name = "next_action_prediction_continuous_bins"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In next step {i} (indexed from 0), the agent transitted to the state s{i} = {state}. Based on your observation and understanding of the agent's behaviour, predict the action a{i} that the RL agent will most likely take at state s{i}.
        The action space dimension is {action_dim}, with each dimension having 10 discrete bins ranging from [-2, 2]: [-2., -1.6), [-1.6, -1.2), [-1.2, -0.8), [-0.8, -0.4), [-0.4, 0.), [0., 0.4), [0.4, 0.8), [0.8, 1.2), [1.2, 1.6), [1.6, 2.]. You only need to predict which bin (indexed from 0 to 9) each element of action a{i} will fall into. Begin with a compact reasoning, followed by a step-by-step prediction for each element of a{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            Action element [i] is a bin index from 0 to 9, where "i" indicates the i-th action element (also indexed from 0).
            Return a list with the following example format,
            ```python
            # For example, if the action dim is 2, and the predicted action element [0] is -1.5 (inside the bin 1: -1.6~-1.2) and element [1] is 1.25 (inside the bin 8: 1.2~1.6), then the predictions would be [1, 8]
            predictions = [1, 8]
            ```
            Note that the provided example needs to be adapted to the current action dim, which is {action_dim}. Please choose only a single integer from 0 to 9 for each action dim, even if multiple outcomes seem possible.
        """
        del task_name

    def render(self, i: int, action_dim: int, state):
        return self.question.format(i=i, action_dim=action_dim, state=state)

class NextActionPredictionContinuousBinsFetch:

    question_name = "next_action_prediction_continuous_bins_fetch"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In next step {i} (indexed from 0), the agent transitted to the state s{i} = {state}. Based on your observation and understanding of the agent's behaviour, predict the action a{i} that the RL agent will most likely take at state s{i}.
        The action space dimension is {action_dim}, with each dimension having 10 discrete bins ranging from [-1, 1]: [-1., -0.8), [-0.8, -0.6), [-0.6, -0.4), [-0.4, -0.2), [-0.2, 0.), [0., 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.]. You only need to predict which bin (indexed from 0 to 9) each element of action a{i} will fall into. Begin with a compact reasoning, followed by a step-by-step prediction for each element of a{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            Action element [i] is a bin index from 0 to 9, where "i" indicates the i-th action element (also indexed from 0).
            Return a list with the following example format,
            ```python
            # For example, if the action dim is 2, and the predicted action element [0] is -0.75 (inside the bin 1: -0.8~-0.6) and element [1] is 0.5 (inside the bin 7: 0.4~0.6), then the predictions would be [1, 7]
            predictions = [1, 7]
            ```
            Note that the provided example needs to be adapted to the current action dim, which is {action_dim}. Please choose only a single integer from 0 to 9 for each action dim, even if multiple outcomes seem possible.
        """
        del task_name

    def render(self, i: int, action_dim: int, state):
        return self.question.format(i=i, action_dim=action_dim, state=state)

class NextActionPredictionContinuousBinsMJPen:

    question_name = "next_action_prediction_continuous_bins_mjpen"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In next step {i} (indexed from 0), the agent transitted to the state s{i} = {state}. Based on your observation and understanding of the agent's behaviour, predict the action a{i} that the RL agent will most likely take at state s{i}.
        The action space dimension is {action_dim}, with each dimension having 10 discrete bins ranging from [-3, 3]: [-3., -2.4), [-2.4, -1.8), [-1.8, -1.2), [-1.2, -0.6), [-0.6, 0.), [0., 0.6), [0.6, 1.2), [1.2, 1.8), [1.8, 2.4), [2.4, 3.]. You only need to predict which bin (indexed from 0 to 9) each element of action a{i} will fall into. Begin with a compact reasoning, followed by a step-by-step prediction for each element of a{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            Action element [i] is a bin index from 0 to 9, where "i" indicates the i-th action element (also indexed from 0).
            Return a list with the following example format,
            ```python
            # For example, if the action dim is 2, and the predicted action element [0] is -2.25 (inside the bin 1: -2.4~-1.8) and element [1] is 1.75 (inside the bin 7: 1.2~1.8), then the predictions would be [1, 7]
            predictions = [1, 7]
            ```
            Note that the provided example needs to be adapted to the current action dim, which is {action_dim}. Please choose only a single integer from 0 to 9 for each action dim, even if multiple outcomes seem possible.
        """
        del task_name

    def render(self, i: int, action_dim: int, state):
        return self.question.format(i=i, action_dim=action_dim, state=state)

class NextActionPredictionContinuousNoBins:

    question_name = "next_action_prediction_continuous_no_bins"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In next step {i} (indexed from 0), the agent transitted to the state s{i} = {state}. Based on your observation and understanding of the agent's behaviour, predict the action a{i} that the RL agent will most likely take at state s{i}.
        The action space dimension is {action_dim}, with each dimension ranging from [-2, 2]. Begin with a compact reasoning, followed by a step-by-step prediction for each element of a{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            Action element [i] is a real value from the action range above, where "i" indicates the i-th action element (also indexed from 0).
            Return a list with the following example format,
            ```python
            # For example, if the action dim is 2, and the predicted action element [0] is -1.52 and element [1] is 1.25, then the predictions would be [-1.52, 1.25]
            predictions = [-1.52, 1.25]
            ```
            Note that the provided example needs to be adapted to the current action dim, which is {action_dim}. Predict a real value with up to two decimal places for each action dim.
        """
        del task_name

    def render(self, i: int, action_dim: int, state):
        return self.question.format(i=i, action_dim=action_dim, state=state)

class NextActionPredictionContinuousNoBinsFetch:

    question_name = "next_action_prediction_continuous_no_bins_fetch"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In next step {i} (indexed from 0), the agent transitted to the state s{i} = {state}. Based on your observation and understanding of the agent's behaviour, predict the action a{i} that the RL agent will most likely take at state s{i}.
        The action space dimension is {action_dim}, with each dimension ranging from [-1, 1]. Begin with a compact reasoning, followed by a step-by-step prediction for each element of a{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            Action element [i] is a real value from the action range above, where "i" indicates the i-th action element (also indexed from 0).
            Return a list with the following example format,
            ```python
            # For example, if the action dim is 2, and the predicted action element [0] is -0.52 and element [1] is 0.25, then the predictions would be [-0.52, 0.25]
            predictions = [-0.52, 0.25]
            ```
            Note that the provided example needs to be adapted to the current action dim, which is {action_dim}. Predict a real value with up to two decimal places for each action dim.
        """
        del task_name

    def render(self, i: int, action_dim: int, state):
        return self.question.format(i=i, action_dim=action_dim, state=state)

class NextActionPredictionContinuousNoBinsMJPen:

    question_name = "next_action_prediction_continuous_no_bins_mjpen"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In next step {i} (indexed from 0), the agent transitted to the state s{i} = {state}. Based on your observation and understanding of the agent's behaviour, predict the action a{i} that the RL agent will most likely take at state s{i}.
        The action space dimension is {action_dim}, with each dimension ranging from [-3, 3]. Begin with a compact reasoning, followed by a step-by-step prediction for each element of a{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            Action element [i] is a real value from the action range above, where "i" indicates the i-th action element (also indexed from 0).
            Return a list with the following example format,
            ```python
            # For example, if the action dim is 2, and the predicted action element [0] is -2.52 and element [1] is 2.15, then the predictions would be [-2.52, 2.15]
            predictions = [-2.52, 2.15]
            ```
            Note that the provided example needs to be adapted to the current action dim, which is {action_dim}. Predict a real value with up to two decimal places for each action dim.
        """
        del task_name

    def render(self, i: int, action_dim: int, state):
        return self.question.format(i=i, action_dim=action_dim, state=state)

# class NextActionPredictionVicuna:  # repeated

#     question_name = "next_action_prediction_vicuna"

#     def __init__(self, task_name=None) -> None:
#         self.question = """
#         In next step {i} (indexed from 0), the agent transitted to the state s{i} = {state}. Based on your observation and understanding of the agent's behaviour, can you predict the action a{i} (an integer from the given range) the RL agent will most likely take at step {i}? Please first provide a compact reasoning before your answer to the action choice. Think step by step and use the following template in your provided answer:

#         1. [Reasoning]:
#         2. [Prediction]:
#         3. [Formatting]:
#             Return a list with the following example format,
#             ```python
#             # final action choice is 0
#             action_choice = [0]
#             ```
#         Please choose only one action, even if multiple actions seem possible.
#         """
#         del task_name

#     def render(self, i: int, state):
#         return self.question.format(i=i, state=state)

class NextActionPredictionLlama3:

    question_name = "next_action_prediction_llama3"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In next step {i} (indexed from 0), the agent transitted to the state s{i} = {state}. Based on your observation and understanding of the agent's behaviour, can you predict the action a{i} (an integer from the given range) the RL agent will most likely take at step {i}? Please first provide a compact reasoning before your answer to the action choice. Think step by step and use the following template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            Return a list with the following example format,
            ```python
            # final action choice is 0
            action_choice = [0]
            ```
        Please choose only one action, even if multiple actions seem possible.
        """
        del task_name

    def render(self, i: int, state):
        return self.question.format(i=i, state=state)

class NextActionPredictionWithTool:  ## In Trial and Remove Later

    question_name = "next_action_prediction_tool"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In the next step {i} (indexed from 0), the agent transitioned to the state s{i} = {state}.
        Statistical Analysis Insight: A regression model using numerical transitions statistics (state, action, reward) predicts the next action as {option}.

        Q: Given the agent's past behavior and the current state, predict the next action a{i}.

        Let's think step by step and use the following template in your provided answer (be concise):

        1. [Reasoning]: Explain the rationale behind your prediction based on the agent's historical transitions and the given state. Discuss whether the statistical prediction aligns with the agent's behavior or if there are discrepancies that lead to a different conclusion.
        2. [Prediction]: State the action you predict the agent will take next.
        3. >>Final action choice: []

        Please choose only one action, even if multiple actions seem possible.
        """
        del task_name

    def render(self, i: int, state, option):
        return self.question.format(i=i, state=state, option=option)

##### LastActionPrediction

"""
In next step {i} (indexed from 0), the agent transitted to the state s{i} = {state}. From that state s{i}, it then moved to state s{k} = {next_state} after a subsequent action a{i} at step {i}. Based on your observation and understanding of the agent's behaviour patterns up to that point, can you predict the action a{i} (an integer from the given range) that the RL agent has most likely taken at step {i} to arrive at state s{k} = {next_state}? Please first provide a compact reasoning before your answer to the action choice. Think step by step and use the following template in your provided answer:
"""

class LastActionPrediction:

    question_name = "last_action_prediction"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In subsequent step {i} (indexed from 0), the state was s{i} = {state}. Then the agent took an action a{i} and the state transitted to s{k} = {next_state}, predict the taken action a{i} (an integer from the given range). Think step by step and use the following template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. >>Final action choice: []

        Please choose only one action, even if multiple actions seem possible.
        """
        del task_name

    def render(self, i: int, k: int, state, next_state):
        return self.question.format(i=i, k=k, state=state, next_state=next_state)

# class LastActionPredictionVicuna:  # repeated

#     question_name = "last_action_prediction_vicuna"

#     def __init__(self, task_name=None) -> None:
#         self.question = """
#         In step {i} (indexed from 0), the state was s{i} = {state}. Then the agent took an action a{i} and the state transitted to s{k} = {next_state}, predict the taken action a{i} (an integer from the given range). Think step by step and use the following template in your provided answer:

#         1. [Reasoning]:
#         2. [Prediction]:
#         3. [Formatting]:
#             Return a list with the following example format,
#             ```python
#             # final action choice is 0
#             action_choice = [0]
#             ```
#         Please choose only one action, even if multiple actions seem possible.
#         """
#         del task_name

#     def render(self, i: int, k: int, state, next_state):
#         return self.question.format(i=i, k=k, state=state, next_state=next_state)

class LastActionPredictionLlama3:

    question_name = "last_action_prediction_llama3"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In step {i} (indexed from 0), the state was s{i} = {state}. Then the agent took an action a{i} and the state transitted to s{k} = {next_state}, predict the taken action a{i} (an integer from the given range). Think step by step and use the following template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            Return a list with the following example format,
            ```python
            # final action choice is 0
            action_choice = [0]
            ```
        Please choose only one action, even if multiple actions seem possible.
        """
        del task_name

    def render(self, i: int, k: int, state, next_state):
        return self.question.format(i=i, k=k, state=state, next_state=next_state)

class LastActionPredictionContinuousBins:

    question_name = "last_action_prediction_continuous_bins"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In next step {i} (indexed from 0), the agent's state was s{i} = {state}. Then the agent took an action a{i} and the state transitted to s{k} = {next_state}. Based on your observation and understanding of the agent's behaviour, predict the action a{i} taken by the RL agent.
        The action space dimension is {action_dim}, with each dimension having 10 discrete bins ranging from [-2, 2]: [-2., -1.6), [-1.6, -1.2), [-1.2, -0.8), [-0.8, -0.4), [-0.4, 0.), [0., 0.4), [0.4, 0.8), [0.8, 1.2), [1.2, 1.6), [1.6, 2.]. You only need to predict which bin (indexed from 0 to 9) each element of action a{i} will fall into. Begin with a compact reasoning, followed by a step-by-step prediction for each element of a{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            Action element [i] is a bin index from 0 to 9, where "i" indicates the i-th action element (also indexed from 0).
            Return a list with the following example format,
            ```python
            # For example, if the action dim is 2, and the predicted action element [0] is -1.5 (inside the bin 1: -1.6~-1.2) and element [1] is 1.25 (inside the bin 8: 1.2~1.6), then the predictions would be [1, 8]
            predictions = [1, 8]
            ```
            Note that the provided example needs to be adapted to the current action dim, which is {action_dim}. Please choose only a single integer from 0 to 9 for each action dim, even if multiple outcomes seem possible.
        """
        del task_name

    def render(self, i: int, k: int, action_dim: int, state, next_state):
        return self.question.format(i=i, k=k, action_dim=action_dim, state=state, next_state=next_state)

class LastActionPredictionContinuousBinsFetch:

    question_name = "last_action_prediction_continuous_bins_fetch"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In next step {i} (indexed from 0), the agent's state was s{i} = {state}. Then the agent took an action a{i} and the state transitted to s{k} = {next_state}. Based on your observation and understanding of the agent's behaviour, predict the action a{i} taken by the RL agent.
        The action space dimension is {action_dim}, with each dimension having 10 discrete bins ranging from [-1, 1]: [-1., -0.8), [-0.8, -0.6), [-0.6, -0.4), [-0.4, -0.2), [-0.2, 0.), [0., 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.]. You only need to predict which bin (indexed from 0 to 9) each element of action a{i} will fall into. Begin with a compact reasoning, followed by a step-by-step prediction for each element of a{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            Action element [i] is a bin index from 0 to 9, where "i" indicates the i-th action element (also indexed from 0).
            Return a list with the following example format,
            ```python
            # For example, if the action dim is 2, and the predicted action element [0] is -0.75 (inside the bin 1: -0.8~-0.6) and element [1] is 0.5 (inside the bin 7: 0.4~0.6), then the predictions would be [1, 7]
            predictions = [1, 7]
            ```
            Note that the provided example needs to be adapted to the current action dim, which is {action_dim}. Please choose only a single integer from 0 to 9 for each action dim, even if multiple outcomes seem possible.
        """
        del task_name

    def render(self, i: int, k: int, action_dim: int, state, next_state):
        return self.question.format(i=i, k=k, action_dim=action_dim, state=state, next_state=next_state)

class LastActionPredictionContinuousBinsMJPen:

    question_name = "last_action_prediction_continuous_bins_mjpen"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In next step {i} (indexed from 0), the agent's state was s{i} = {state}. Then the agent took an action a{i} and the state transitted to s{k} = {next_state}. Based on your observation and understanding of the agent's behaviour, predict the action a{i} taken by the RL agent.
        The action space dimension is {action_dim}, with each dimension having 10 discrete bins ranging from [-3, 3]: [-3., -2.4), [-2.4, -1.8), [-1.8, -1.2), [-1.2, -0.6), [-0.6, 0.), [0., 0.6), [0.6, 1.2), [1.2, 1.8), [1.8, 2.4), [2.4, 3.]. You only need to predict which bin (indexed from 0 to 9) each element of action a{i} will fall into. Begin with a compact reasoning, followed by a step-by-step prediction for each element of a{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            Action element [i] is a bin index from 0 to 9, where "i" indicates the i-th action element (also indexed from 0).
            Return a list with the following example format,
            ```python
            # For example, if the action dim is 2, and the predicted action element [0] is -2.25 (inside the bin 1: -2.4~-1.8) and element [1] is 1.75 (inside the bin 7: 1.2~1.8), then the predictions would be [1, 7]
            predictions = [1, 7]
            ```
            Note that the provided example needs to be adapted to the current action dim, which is {action_dim}. Please choose only a single integer from 0 to 9 for each action dim, even if multiple outcomes seem possible.
        """
        del task_name

    def render(self, i: int, k: int, action_dim: int, state, next_state):
        return self.question.format(i=i, k=k, action_dim=action_dim, state=state, next_state=next_state)


class LastActionPredictionContinuousNoBins:

    question_name = "last_action_prediction_continuous_no_bins"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In next step {i} (indexed from 0), the agent's state was s{i} = {state}. Then the agent took an action a{i} and the state transitted to s{k} = {next_state}. Based on your observation and understanding of the agent's behaviour, predict the action a{i} taken by the RL agent.
        The action space dimension is {action_dim}, with each dimension ranging from [-2, 2]. Begin with a compact reasoning, followed by a step-by-step prediction for each element of a{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            Action element [i] is a real value from the action range above, where "i" indicates the i-th action element (also indexed from 0).
            Return a list with the following example format,
            ```python
            # For example, if the action dim is 2, and the predicted action element [0] is -1.52 and element [1] is 1.25, then the predictions would be [-1.52, 1.25]
            predictions = [-1.52, 1.25]
            ```
            Note that the provided example needs to be adapted to the current action dim, which is {action_dim}. Predict a real value with up to two decimal places for each action dim.
        """
        del task_name

    def render(self, i: int, k: int, action_dim: int, state, next_state):
        return self.question.format(i=i, k=k, action_dim=action_dim, state=state, next_state=next_state)

class LastActionPredictionContinuousNoBinsFetch:

    question_name = "last_action_prediction_continuous_no_bins_fetch"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In next step {i} (indexed from 0), the agent's state was s{i} = {state}. Then the agent took an action a{i} and the state transitted to s{k} = {next_state}. Based on your observation and understanding of the agent's behaviour, predict the action a{i} taken by the RL agent.
        The action space dimension is {action_dim}, with each dimension ranging from [-1, 1]. Begin with a compact reasoning, followed by a step-by-step prediction for each element of a{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            Action element [i] is a real value from the action range above, where "i" indicates the i-th action element (also indexed from 0).
            Return a list with the following example format,
            ```python
            # For example, if the action dim is 2, and the predicted action element [0] is -0.52 and element [1] is 0.25, then the predictions would be [-0.52, 0.25]
            predictions = [-0.52, 0.25]
            ```
            Note that the provided example needs to be adapted to the current action dim, which is {action_dim}. Predict a real value with up to two decimal places for each action dim.
        """
        del task_name

    def render(self, i: int, k: int, action_dim: int, state, next_state):
        return self.question.format(i=i, k=k, action_dim=action_dim, state=state, next_state=next_state)

class LastActionPredictionContinuousNoBinsMJPen:

    question_name = "last_action_prediction_continuous_no_bins_mjpen"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In next step {i} (indexed from 0), the agent's state was s{i} = {state}. Then the agent took an action a{i} and the state transitted to s{k} = {next_state}. Based on your observation and understanding of the agent's behaviour, predict the action a{i} taken by the RL agent.
        The action space dimension is {action_dim}, with each dimension ranging from [-3, 3]. Begin with a compact reasoning, followed by a step-by-step prediction for each element of a{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            Action element [i] is a real value from the action range above, where "i" indicates the i-th action element (also indexed from 0).
            Return a list with the following example format,
            ```python
            # For example, if the action dim is 2, and the predicted action element [0] is -2.52 and element [1] is 2.15, then the predictions would be [-2.52, 2.15]
            predictions = [-2.52, 2.15]
            ```
            Note that the provided example needs to be adapted to the current action dim, which is {action_dim}. Predict a real value with up to two decimal places for each action dim.
        """
        del task_name

    def render(self, i: int, k: int, action_dim: int, state, next_state):
        return self.question.format(i=i, k=k, action_dim=action_dim, state=state, next_state=next_state)

#####
class NextRewardPrediction:

    question_name = "next_reward_prediction"  # where is the use-case?

    def __init__(self, task_name=None) -> None:
        self.question = """
        In next step {i} (indexed from 0), the agent transitted to the state s{i} = {state}. Based on your observation and understanding of the agent's behaviour, can you predict the action a{i} (an integer from the given range) the RL agent will most likely take at step {i}? Please first provide a compact reasoning before your answer to the action choice. Think step by step and use the following template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. >>Final action choice: []

        Please choose only one action, even if multiple actions seem possible.
        """
        del task_name

    def render(self, i: int, state):
        return self.question.format(i=i, state=state)


##### ArgueAction
"""
        In the next step {i} (indexed from 0), the agent transitioned to the state s{i} = {state}. In this state s{i}, the agent chose to take action a{i} = {action}. Based on your observation and understanding of the agent's behaviour up to this point, you need to reason over why the agent took action a{i} at state s{i}. It's allowed to question the correctness of the action a{i} when you've deduced that the action may be actually incorrect based on your careful reasoning. Finally you need to answer if you wanna question the correctness of agent's action a{i}. Please first provide a compact reasoning before your answer. Think step by step and use the following template in your provided answer:

        1. [Reasoning]: Summarize why the agent chose action a{i}.
        2. [Justification]: Explain whether the action seems appropriate given the state s{i} and whether there are reasons to doubt its correctness.
        3. [Formatting]:
            Final vote is [True] (argue for the agent's action) or [False] (argue against the agent's action)
            return a list with the following example format,
            ```python
            # final vote is to argue against the agent's action
            final_vote = [False]
            ```
        Please choose either True or False, even in cases of uncertainty or multiple possibilities.
        """

"""
After your evaluation, determine if you should question the correctness of agent's action a{i}. Think step by step and use the following template in your provided answer:

[Reasoning]: Detail your understanding of why the agent chose action a{i}.
"""

"""
 3. **Vote**:
        Make a definitive choice:
        - [True] if you support the agent's action as being correct.
        - [False] if you believe the agent's action was incorrect.
        Provide your vote in the format shown:
        ```python
        # Example: voting against the agent's action
        final_vote = [False]
"""

class ArgueAction:

    question_name = "argue_action"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In the next step {i} (indexed from 0), the agent transitioned to the state s{i} = {state}. In this state s{i}, the agent chose to take action a{i} = {action}. Based on your observation and understanding of the agent's behaviour up to this point, critically evaluate the rationale behind the agent's choice of action a{i} in state s{i}. You're encouraged to scrutinize the correctness of the action a{i}, especially if your analysis suggests that the action might be flawed or suboptimal.

        After your evaluation, determine whether to accept or reject the agent's action a{i}. Think step by step and use the following template in your provided answer:

        1. [Reasoning]:
        2. [Justification]: Critique whether the action is correct given the historical context and the state s{i}, and whether there are any reasons to doubt its correctness.
        3. [Formatting]:
            Final vote is [True] (i.e., argue for the agent's action) or [False] (i.e., argue against the agent's action)
            Return a list with the following example format,
            ```python
            # final vote is to argue against the agent's action
            final_vote = [False]
            ```
        Please choose either True or False, even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, i: int, state, action):
        return self.question.format(i=i, state=state, action=action)

class ArgueCorrectAction:

    question_name = "argue_correct_action"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In the next step {i} (indexed from 0), the agent transitioned to the state s{i} = {state}. In this state s{i}, the agent chose to take action a{i} = {action}. Based on your observation and understanding of the agent's behaviour up to this point, critically evaluate the rationale behind the agent's choice of action a{i} in state s{i}. You're encouraged to scrutinize the correctness of the action a{i}, especially if your analysis suggests that the action might be flawed or suboptimal.

        After your evaluation, determine whether to accept or reject the agent's action a{i}. Think step by step and use the following template in your provided answer:

        1. [Reasoning]:
        2. [Justification]: Critique whether the action is correct given the historical context and the state s{i} and whether there are any reasons to doubt its correctness.
        3. [Formatting]:
            Final vote is [True] (argue for the agent's action) or [False] (argue against the agent's action)
            Return a list with the following example format,
            ```python
            # final vote is to argue against the agent's action
            final_vote = [False]
            ```
        Please choose either True or False, even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, i: int, state, action):
        return self.question.format(i=i, state=state, action=action)

class ArgueWrongAction:

    question_name = "argue_wrong_action"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In the next step {i} (indexed from 0), the agent transitioned to the state s{i} = {state}. In this state s{i}, the agent chose to take action a{i} = {action}. Based on your observation and understanding of the agent's behaviour up to this point, critically evaluate the rationale behind the agent's choice of action a{i} in state s{i}. You're encouraged to scrutinize the correctness of the action a{i}, especially if your analysis suggests that the action might be flawed or suboptimal.

        After your evaluation, determine whether to accept or reject the agent's action a{i}. Think step by step and use the following template in your provided answer:

        1. [Reasoning]:
        2. [Justification]: Critique whether the action is correct given the historical context and the state s{i} and whether there are any reasons to doubt its correctness.
        3. [Formatting]:
            Final vote is [True] (argue for the agent's action) or [False] (argue against the agent's action)
            Return a list with the following example format,
            ```python
            # final vote is to argue against the agent's action
            final_vote = [False]
            ```
        Please choose either True or False, even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, i: int, state, action):
        return self.question.format(i=i, state=state, action=action)


class ArgueActionLlama3:

    question_name = "argue_action_llama3"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In the next step {i} (indexed from 0), the agent transitioned to the state s{i} = {state}. In this state s{i}, the agent chose to take action a{i} = {action}. Based on your observation and understanding of the agent's behaviour up to this point, critically evaluate the rationale behind the agent's choice of action a{i} in state s{i}. You're encouraged to scrutinize the correctness of the action a{i}, especially if your analysis suggests that the action might be flawed or suboptimal.

        After your evaluation, determine whether to accept or reject the agent's action a{i}. Think step by step and use the following template in your provided answer:

        1. [Reasoning]:
        2. [Justification]: Critique whether the action is correct given the historical context and the state s{i} and whether there are any reasons to doubt its correctness.
        3. [Formatting]:
            Final vote is [True] (argue for the agent's action) or [False] (argue against the agent's action)
            Return a list with the following example format,
            ```python
            # final vote is to argue against the agent's action
            final_vote = [False]
            ```
        Please choose either True or False, even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, i: int, state, action):
        return self.question.format(i=i, state=state, action=action)


class ArgueActionVicuna:

    question_name = "argue_action_vicuna"

    def __init__(self, task_name=None) -> None:
        self.question = """
        In the next step {i} (indexed from 0), the agent transitioned to the state s{i} = {state}. In this state s{i}, the agent chose to take action a{i} = {action}. Based on your observation and understanding of the agent's behaviour up to this point, critically evaluate the rationale behind the agent's choice of action a{i} in state s{i}. You're encouraged to scrutinize the correctness of the action a{i}, especially if your analysis suggests that the action might be flawed or suboptimal.

        After your evaluation, determine whether to accept or reject the agent's action a{i}. Think step by step and use the following template in your provided answer:

        1. [Reasoning]:
        2. [Justification]: Critique whether the action is correct given the historical context and the state s{i} and whether there are any reasons to doubt its correctness.
        3. [Formatting]:
            Final vote is [True] (argue for the agent's action) or [False] (argue against the agent's action)
            Return a list with the following example format,
            ```python
            # final vote is to argue against the agent's action
            final_vote = [False]
            ```
        Please choose either True or False, even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, i: int, state, action):
        return self.question.format(i=i, state=state, action=action)

##### NextStatePrediction
"""
        1. [Reasoning]:
        2. [Prediction]:
        3. >>Final state status:
            State element [i]: INC (increase) or DEC (decrease), where "i" indicates the index of the state element (indexed from 0)
            return a list with the following example format,
            ```python
            # element [0] increases and element [1] decreases
            predictions = ["INC", "DEC"]
            ```
        Please choose either INC or DEC, even in cases of uncertainty or multiple possibilities.
"""

class NextStatePredictionMoreOptions:

    question_name = "next_state_prediction_more_options"

    def __init__(self, task_name=None) -> None:
        self.question = """
        Using the history of states, actions, and rewards up to step {j} (indexed from 0), predict the next state s{i} (the agent will transition to) that follows from the current state s{j} = {state}, action a{j} = {action}, and reward r{j} = {reward} received at step {j}. You only need to deduce whether each element of the next state s{i} increases (symbolized as "INC"), decreases (symbolized as "DEC"), or remains unchanged (symbolized as "UNCH") compared to the current state s{j} = {state} after taking action a{j}. The state element can stay unchanged if the difference between elements of two successive states is less than a threshold of 1e-4. Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction. Begin with a compact reasoning, followed by a step-by-step prediction for each element of s{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            State element [i] INC (increase), DEC (decrease), or UNCH (unchange), where "i" indicates the index of the state element (indexed from 0).
            Return a list with the following example format,
            ```python
            # element [0] increases, element [1] decreases, and element [2] remains unchanged
            predictions = ["INC", "DEC", "UNCH"]
            ```
            Please ensure each state element prediction explicitly states either "INC", "DEC", or "UNCH", even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, j: int, i: int, state, action, reward):
        return self.question.format(j=j, i=i, state=state, action=action, reward=reward)

class NextStatePrediction:

    question_name = "next_state_prediction"

    def __init__(self, task_name=None) -> None:
        self.question = """
        Using the history of states, actions, and rewards up to step {j} (indexed from 0), predict the next state s{i} (the agent will transition to) that follows from the current state s{j} = {state}, action a{j} = {action}, and reward r{j} = {reward} received at step {j}. You only need to deduce whether each element of the next state s{i} increases (symbolized as "INC") or decreases (symbolized as "DEC") compared to the current state s{j} = {state} after taking action a{j}. Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction. Begin with a compact reasoning, followed by a step-by-step prediction for each element of s{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            State element [i] INC (increase) or DEC (decrease), where "i" indicates the index of the state element (indexed from 0).
            Return a list with the following example format,
            ```python
            # element [0] increases and element [1] decreases
            predictions = ["INC", "DEC"]
            ```
            Please choose either INC or DEC, even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, j: int, i: int, state, action, reward):
        return self.question.format(j=j, i=i, state=state, action=action, reward=reward)

class NextStatePredictionFetchPart:

    question_name = "next_state_prediction_fetch_part"

    def __init__(self, task_name=None) -> None:
        self.question = """
        Using the history of states, actions, and rewards up to step {j} (indexed from 0), predict the end effector 3-d positions ee_pos{i} and block 3-d positions block_pos{i} in the next state s{i} (the agent will transition to) that follows from the current state s{j} = {state}, action a{j} = {action}, and reward r{j} = {reward} received at step {j}. You only need to deduce whether each element of ee_pos{i} and block_pos{i} increases (symbolized as "INC"), decreases (symbolized as "DEC"), or remains unchanged (symbolized as "UNCH") compared to the current end effector positions ee_pos{j} = {ee_pos} and block positions block_pos{j} = {block_pos} in state s{j} after taking action a{j}. The state element can stay unchanged if the difference between elements of two successive states is less than a threshold of 1e-4. Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction. Begin with a compact reasoning, followed by a step-by-step prediction for each element of ee_pos{i} and block_pos{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            State element [i] INC (increase), DEC (decrease), or UNCH (unchange), where "i" indicates the index of the state element (indexed from 0) and is indexed from 0 to 5.
            Return a list with the following example format,
            ```python
            # element [0] increases, element [1] decreases, element [2] remains unchanged, element [3] remains unchanged, element [4] decreases, and element [5] increases
            predictions = ["INC", "DEC", "UNCH", "UNCH", "DEC", "INC"]
            ```
            Please ensure each prediction explicitly states either "INC", "DEC", or "UNCH", even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, j: int, i: int, state, ee_pos, block_pos, action, reward):
        return self.question.format(j=j, i=i, state=state, ee_pos=ee_pos, block_pos=block_pos, action=action, reward=reward)

class NextStatePredictionFetch:

    question_name = "next_state_prediction_fetch"

    def __init__(self, task_name=None) -> None:
        self.question = """
        Using the history of states, actions, and rewards up to step {j} (indexed from 0), predict the next state s{i} (the agent will transition to) that follows from the current state s{j} = {state}, action a{j} = {action}, and reward r{j} = {reward} received at step {j}. You only need to deduce whether each element of the next state s{i} increases (symbolized as "INC"), decreases (symbolized as "DEC"), or remains unchanged (symbolized as "UNCH") compared to the current state s{j} after taking action a{j}. The state element can stay unchanged if the difference between elements of two successive states is less than a threshold of 1e-4. Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction. Begin with a compact reasoning, followed by a step-by-step prediction for each element of s{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            State element [i] INC (increase), DEC (decrease), or UNCH (unchange), where "i" indicates the index of the state element (indexed from 0).
            Return a list with the following example format,
            ```python
            # element [0] increases, element [1] decreases, and element [2] remains unchanged
            predictions = ["INC", "DEC", "UNCH"]
            ```
            Please ensure each state element prediction explicitly states either "INC", "DEC", or "UNCH", even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, j: int, i: int, state, action, reward):
        return self.question.format(j=j, i=i, state=state, action=action, reward=reward)

class NextStatePredictionWithDynamics:  # IN TRIAL

    question_name = "next_state_prediction"

    def __init__(self, task_name=None) -> None:
        self.question = """
        Predict the next state s{i} (the agent will transition to) that follows from the current state s{j} = {state}, action a{j} = {action}, and reward r{j} = {reward} received at step {j} (indexed from 0). You only need to deduce whether each element of the next state s{i} increases (symbolized as "INC") or decreases (symbolized as "DEC") compared to the current state s{j} = {state} after taking action a{j}. Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction. Begin with a compact reasoning, followed by a step-by-step prediction for each element of s{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
            Explicitly follow the provided transition dynamics (equation in python)
            ```python
            velocity(t+1) = velocity(t) + (action - 1) * force - np.cos(3 * position(t)) * gravity
            position(t+1) = position(t) + velocity(t+1)
            ```
            to calculate for the right answer step by step.
        3. [Formatting]:
            State element [i] INC (increase) or DEC (decrease), where "i" indicates the index of the state element (indexed from 0).
            Return a list with the following example format,
            ```python
            # element [0] increases and element [1] decreases
            predictions = ["INC", "DEC"]
            ```
            Please choose either INC or DEC, even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, j: int, i: int, state, action, reward):
        return self.question.format(j=j, i=i, state=state, action=action, reward=reward)

"""
        Using the history of states, actions, and rewards up to step {j} (indexed from 0), predict the values of each element of the next state s{i} based on the current state s{j} = {state}, action a{j} = {action}, and reward r{j} = {reward} received at step {j}. Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction. Provide a compact reasoning followed by a step-by-step prediction for each element of s{i} using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            State element [i] = value, where "i" indicates the index of the state element (indexed from 0) and "value" is the predicted integer value of the state element.
            Return a list with the following example format:
            ```python
            # element [0] is 1, element [1] is 3, and element [2] is 5 the values of the next state elements
            next_state = [value_0, value_1, value_2]
            ```
            Please ensure each state element prediction provides an integer value.
        """

class NextStatePredictionDiscrete:

    question_name = "next_state_prediction_discrete"

    def __init__(self, task_name=None) -> None:
        self.question = """
        Using the history of states, actions, and rewards up to step {j} (indexed from 0), predict the values of each element of the next state s{i} based on the current state s{j} = {state}, action a{j} = {action}, and reward r{j} = {reward} received at step {j}. Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction. Provide a compact reasoning followed by a step-by-step prediction for each element of s{i} using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            State element [i] = value, where "i" indicates the index of the state element (indexed from 0) and "value" is the predicted integer value of the state element.
            Return a list with the following example format:
            ```python
            # element [0] is 1, element [1] is 3, and element [2] is 5 the values of the next state elements
            next_state = [value_0, value_1, value_2]
            ```
            Please ensure each state element prediction provides an integer value.
        """
        del task_name

    def render(self, j: int, i: int, state, action, reward):
        return self.question.format(j=j, i=i, state=state, action=action, reward=reward)


class NextStatePredictionDiscreteDelta:  # to be decided on what to predict

    question_name = "next_state_prediction_discrete"

    def __init__(self, task_name=None) -> None:
        self.question = """
        Using the history of states, actions, and rewards up to step {j} (indexed from 0), predict the next state s{i} (the agent will transition to) that follows from the current state s{j} = {state}, action a{j} = {action}, and reward r{j} = {reward} received at step {j}. You only need to deduce whether each element of the next state s{i} increases (symbolized as "INC") by one, decreases (symbolized as "DEC") by one, or remains unchanged (symbolized as "UNCH") compared to the current state s{j} = {state} after taking action a{j}. Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction. Begin with a compact reasoning, followed by a step-by-step prediction for each element of s{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            State element [i] INC (increase) by one, DEC (decrease) by one, or UNCH (unchange), where "i" indicates the index of the state element (indexed from 0).
            Return a list with the following example format,
            ```python
            # predicted element [0] increases by one, element [1] decreases by one, and element [2] remains unchanged
            predictions = ["INC", "DEC", "UNCH"]
            ```
            Please ensure each state element prediction explicitly states either "INC", "DEC", or "UNCH", even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, j: int, i: int, state, action, reward):
        return self.question.format(j=j, i=i, state=state, action=action, reward=reward)

class NextStatePredictionMoreOptionsLlama3:

    question_name = "next_state_prediction_more_options_llama3"

    def __init__(self, task_name=None) -> None:
        self.question = """
        Using the history of states, actions, and rewards up to step {j} (indexed from 0), predict the next state s{i} (the agent will transition to) that follows from the current state s{j} = {state}, action a{j} = {action}, and reward r{j} = {reward} received at step {j}. You only need to deduce whether each element of the next state s{i} increases (symbolized as "INC"), decreases (symbolized as "DEC"), or remains unchanged (symbolized as "UNCH") compared to the current state s{j} = {state} after taking action a{j}. The state element can stay unchanged if the difference between elements of two successive states is less than a threshold of 1e-4. Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction. Begin with a compact reasoning, followed by a step-by-step prediction for each element of s{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            State element [i] INC (increase), DEC (decrease), or UNCH (unchange), where "i" indicates the index of the state element (indexed from 0).
            Return a list with the following example format,
            ```python
            # element [0] increases, element [1] decreases, and element [2] remains unchanged
            predictions = ["INC", "DEC", "UNCH"]
            ```
            Please ensure each state element prediction explicitly states either "INC", "DEC", or "UNCH", even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, j: int, i: int, state, action, reward):
        return self.question.format(j=j, i=i, state=state, action=action, reward=reward)

class NextStatePredictionLlama3:

    question_name = "next_state_prediction_llama3"

    def __init__(self, task_name=None) -> None:
        self.question = """
        Using the history of states, actions, and rewards up to step {j} (indexed from 0), predict the next state s{i} (the agent will transition to) that follows from the current state s{j} = {state}, action a{j} = {action}, and reward r{j} = {reward} received at step {j}. You only need to deduce whether each element of the next state s{i} increases (symbolized as "INC") or decreases (symbolized as "DEC") compared to the current state s{j} = {state} after taking action a{j}. Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction. Begin with a compact reasoning, followed by a step-by-step prediction for each element of s{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            State element [i] INC (increase) or DEC (decrease), where "i" indicates the index of the state element (indexed from 0).
            Return a list with the following example format,
            ```python
            # element [0] increases and element [1] decreases
            predictions = ["INC", "DEC"]
            ```
            Please choose either INC or DEC, even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, j: int, i: int, state, action, reward):
        return self.question.format(j=j, i=i, state=state, action=action, reward=reward)

class NextStatePredictionMoreOptionsVicuna:

    question_name = "next_state_prediction_more_options_vicuna"

    def __init__(self, task_name=None) -> None:
        self.question = """
        Using the history of states, actions, and rewards up to step {j} (indexed from 0), predict the next state s{i} (the agent will transition to) that follows from the current state s{j} = {state}, action a{j} = {action}, and reward r{j} = {reward} received at step {j}. You only need to deduce whether each element of the next state s{i} increases (symbolized as "INC"), decreases (symbolized as "DEC"), or remains unchanged (symbolized as "UNCH") compared to the current state s{j} = {state} after taking action a{j}. The state element can stay unchanged if the difference between elements of two successive states is less than a threshold of 1e-4. Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction. Begin with a compact reasoning, followed by a step-by-step prediction for each element of s{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            State element [i] INC (increase), DEC (decrease), or UNCH (unchange), where "i" indicates the index of the state element (indexed from 0).
            Return a list with the following example format,
            ```python
            # element [0] increases, element [1] decreases, and element [2] remains unchanged
            predictions = ["INC", "DEC", "UNCH"]
            ```
            Please ensure each state element prediction explicitly states either "INC", "DEC", or "UNCH", even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, j: int, i: int, state, action, reward):
        return self.question.format(j=j, i=i, state=state, action=action, reward=reward)

class NextStatePredictionVicuna:

    question_name = "next_state_prediction_vicuna"

    def __init__(self, task_name=None) -> None:
        self.question = """
        Using the history of states, actions, and rewards up to step {j} (indexed from 0), predict the next state s{i} (the agent will transition to) that follows from the current state s{j} = {state} and action a{j} = {action} at step {j}. You only need to deduce whether each element of the next state s{i} increases (symbolized as "INC") or decreases (symbolized as "DEC") compared to the current state s{j} = {state} after taking action a{j}. Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction. Begin with a compact reasoning, followed by a step-by-step prediction for each element of s{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            State element [i] INC (increase) or DEC (decrease), where "i" indicates the index of the state element (indexed from 0).
            Return a list with the following example format,
            ```python
            # element [0] increases and element [1] decreases
            predictions = ["INC", "DEC"]
            ```
            Please choose either INC or DEC, even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, j: int, i: int, state, action, reward):
        return self.question.format(j=j, i=i, state=state, action=action, reward=reward)

"""
2. [Prediction]: Step-by-step predictions for each element of s{i}, indicating if it was higher, lower, or unchanged before reaching s{k}.
"""

##### LastStatePrediction

class LastStatePredictionMoreOptions:

    question_name = "last_state_prediction_more_options"

    def __init__(self, task_name=None) -> None:

        self.question = """
        In step {i} (indexed from 0), the state was s{i}. Then the agent took an action a{i} = {action} and the state transitted to s{k} = {next_state} from s{i}. Deduce the previous state s{i} by comparing it to s{k}. You only need to deduce whether each element of the state s{i} was lower, higher, or the same, compared to s{k}, before the action a{i} was taken at state s{i}. The state element can stay unchanged if the difference between elements of two successive states is less than a threshold of 1e-4. Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction. Predict if each element of s{i} increased (symbolized as "INC"), decreased (symbolized as "DEC"), or stayed unchanged (symbolized as "UNCH") to reach s{k}. Begin with a compact reasoning, followed by a step-by-step prediction for each element of s{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            State element [i] INC (increase), DEC (decrease), or UNCH (unchange) to reach the next state, where "i" indicates the index of the state element (indexed from 0).
            Return a list with the following example format,
            ```python
            # element [0] increased, element [1] decreased, and element [2] stayed unchanged to reach the next state
            predictions = ["INC", "DEC", "UNCH"]
            ```
            Please choose either INC, DEC, or UNCH for each element, even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, j: int, i: int, k: int, action, next_state):
        return self.question.format(j=j, i=i, k=k, action=action, next_state=next_state)

class LastStatePrediction:

    question_name = "last_state_prediction"

    def __init__(self, task_name=None) -> None:

        self.question = """
        In step {i} (indexed from 0), the state was s{i}. Then the agent took an action a{i} = {action} and the state transitted to s{k} = {next_state} from s{i}. Deduce the previous state s{i} by comparing it to s{k}. You only need to deduce whether each element of the state s{i} was lower or higher, compared to s{k}, before the action a{i} was taken at state s{i}. Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction. Predict if each element of s{i} increased (symbolized as "INC") or decreased (symbolized as "DEC") to reach s{k}. Begin with a compact reasoning, followed by a step-by-step prediction for each element of s{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            State element [i] INC (increase) or DEC (decrease) to reach the next state, where "i" indicates the index of the state element (indexed from 0).
            Return a list with the following example format,
            ```python
            # element [0] increased and element [1] decreased to reach the next state
            predictions = ["INC", "DEC"]
            ```
            Please choose either INC or DEC, even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, j: int, i: int, k: int, action, next_state):
        return self.question.format(j=j, i=i, k=k, action=action, next_state=next_state)

class LastStatePredictionMoreOptionsLlama3:

    question_name = "last_state_prediction_more_options_llama3"

    def __init__(self, task_name=None) -> None:

        self.question = """
        In step {i} (indexed from 0), the state was s{i}. Then the agent took an action a{i} = {action} and the state transitted to s{k} = {next_state} from s{i}. Deduce the previous state s{i} by comparing it to s{k}. You only need to deduce whether each element of the state s{i} was lower, higher, or the same, compared to s{k}, before the action a{i} was taken at state s{i}. The state element can stay unchanged if the difference between elements of two successive states is less than a threshold of 1e-4. Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction. Predict if each element of s{i} increased (symbolized as "INC"), decreased (symbolized as "DEC"), or stayed unchanged (symbolized as "UNCH") to reach s{k}. Begin with a compact reasoning, followed by a step-by-step prediction for each element of s{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            State element [i] INC (increase), DEC (decrease), or UNCH (unchange) to reach the next state, where "i" indicates the index of the state element (indexed from 0).
            Return a list with the following example format,
            ```python
            # element [0] increased, element [1] decreased, and element [2] stayed unchanged to reach the next state
            predictions = ["INC", "DEC", "UNCH"]
            ```
            Please choose either INC, DEC, or UNCH for each element, even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, j: int, i: int, k: int, action, next_state):
        return self.question.format(j=j, i=i, k=k, action=action, next_state=next_state)

class LastStatePredictionLlama3:

    question_name = "last_state_prediction_llama3"

    def __init__(self, task_name=None) -> None:

        self.question = """
        In step {i} (indexed from 0), the state was s{i}. Then the agent took an action a{i} = {action} and the state transitted to s{k} = {next_state} from s{i}. Deduce the previous state s{i} by comparing it to s{k}. You only need to deduce whether each element of the state s{i} was lower or higher, compared to s{k}, before the action a{i} was taken at state s{i}. Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction. Predict if each element of s{i} increased (symbolized as "INC") or decreased (symbolized as "DEC") to reach s{k}. Begin with a compact reasoning, followed by a step-by-step prediction for each element of s{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            State element [i] INC (increase) or DEC (decrease) to reach the next state, where "i" indicates the index of the state element (indexed from 0).
            Return a list with the following example format,
            ```python
            # element [0] increased and element [1] decreased to reach the next state
            predictions = ["INC", "DEC"]
            ```
            Please choose either INC or DEC, even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, j: int, i: int, k: int, action, next_state):
        return self.question.format(j=j, i=i, k=k, action=action, next_state=next_state)

class LastStatePredictionMoreOptionsVicuna:

    question_name = "last_state_prediction_more_options_vicuna"

    def __init__(self, task_name=None) -> None:

        self.question = """
        In step {i} (indexed from 0), the state was s{i}. Then the agent took an action a{i} = {action} and the state transitted to s{k} = {next_state} from s{i}. Deduce the previous state s{i} by comparing it to s{k}. You only need to deduce whether each element of the state s{i} was lower, higher, or the same, compared to s{k}, before the action a{i} was taken at state s{i}. The state element can stay unchanged if the difference between elements of two successive states is less than a threshold of 1e-4. Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction. Predict if each element of s{i} increased (symbolized as "INC"), decreased (symbolized as "DEC"), or stayed unchanged (symbolized as "UNCH") to reach s{k}. Begin with a compact reasoning, followed by a step-by-step prediction for each element of s{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            State element [i] INC (increase), DEC (decrease), or UNCH (unchange) to reach the next state, where "i" indicates the index of the state element (indexed from 0).
            Return a list with the following example format,
            ```python
            # element [0] increased, element [1] decreased, and element [2] stayed unchanged to reach the next state
            predictions = ["INC", "DEC", "UNCH"]
            ```
            Please choose either INC, DEC, or UNCH for each element, even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, j: int, i: int, k: int, action, next_state):
        return self.question.format(j=j, i=i, k=k, action=action, next_state=next_state)

class LastStatePredictionVicuna:

    question_name = "last_state_prediction_vicuna"

    def __init__(self, task_name=None) -> None:

        self.question = """
        In step {i} (indexed from 0), the state was s{i}. Then the agent took an action a{i} = {action} and the state transitted to s{k} = {next_state} from s{i}. Deduce the previous state s{i} by comparing it to s{k}. You only need to deduce whether each element of the state s{i} was lower or higher, compared to s{k}, before the action a{i} was taken at state s{i}. Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction. Predict if each element of s{i} increased (symbolized as "INC") or decreased (symbolized as "DEC") to reach s{k}. Begin with a compact reasoning, followed by a step-by-step prediction for each element of s{i}, using the template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. [Formatting]:
            State element [i] INC (increase) or DEC (decrease) to reach the next state, where "i" indicates the index of the state element (indexed from 0).
            Return a list with the following example format,
            ```python
            # element [0] increased and element [1] decreased to reach the next state
            predictions = ["INC", "DEC"]
            ```
            Please choose either INC or DEC, even in cases of uncertainty or multiple possibilities.
        """
        del task_name

    def render(self, j: int, i: int, k: int, action, next_state):
        return self.question.format(j=j, i=i, k=k, action=action, next_state=next_state)


NextActionPrediction_CLS = [
    NextActionPrediction,
    NextActionPredictionLlama3,
    NextActionPredictionContinuousBins,
    NextActionPredictionContinuousBinsFetch,
    NextActionPredictionContinuousBinsMJPen,
    NextActionPredictionContinuousNoBins,
    NextActionPredictionContinuousNoBinsFetch,
    NextActionPredictionContinuousNoBinsMJPen,
    ]

LastActionPrediction_CLS = [
    LastActionPrediction,
    LastActionPredictionLlama3,  # repeated
    LastActionPredictionContinuousBins,
    LastActionPredictionContinuousBinsFetch,
    LastActionPredictionContinuousBinsMJPen,
    LastActionPredictionContinuousNoBins,
    LastActionPredictionContinuousNoBinsFetch,
    LastActionPredictionContinuousNoBinsMJPen,
    ]

NextStatePrediction_CLS = [
    NextStatePrediction,
    NextStatePredictionMoreOptions,
    NextStatePredictionVicuna,
    NextStatePredictionMoreOptionsVicuna,
    NextStatePredictionLlama3,
    NextStatePredictionMoreOptionsLlama3
    ]

LastStatePrediction_CLS = [
    LastStatePrediction,
    LastStatePredictionMoreOptions,
    LastStatePredictionVicuna,
    LastStatePredictionMoreOptionsVicuna,
    LastStatePredictionLlama3,
    LastStatePredictionMoreOptionsLlama3,
    ]

ArgueAction_CLS = [
    ArgueAction,
    ArgueActionVicuna,
    ArgueActionLlama3
    ]

QUEST_CLS_POOL = {
    "NextActionPrediction_CLS": NextActionPrediction_CLS,
    "LastActionPrediction_CLS": LastActionPrediction_CLS,
    "NextStatePrediction_CLS": NextStatePrediction_CLS,
    "LastStatePrediction_CLS": LastStatePrediction_CLS,
    "ArgueAction_CLS": ArgueAction_CLS,
    }

# def get_quest_cls(question_name):
#     mapping = {
#         "NextAction": "NextActionPrediction_CLS",
#         "LastAction": "LastActionPrediction_CLS",
#         "NextState": "NextStatePrediction_CLS",
#         "LastState": "LastStatePrediction_CLS",
#         "ArgueAction": "ArgueAction_CLS"
#     }

#     quest_cls_name = next((cls_name for key, cls_name in mapping.items() if key in question_name), None)
#     if quest_cls_name:
#         quest_cls_list = QUEST_CLS_POOL.get(quest_cls_name, [])
#         for quest_cls in quest_cls_list:
#             if quest_cls.question_name == question_name:
#                 return quest_cls

#     return None

def get_quest_cls(quest_cls_name, question_name):
    quest_cls_list = QUEST_CLS_POOL.get(quest_cls_name, [])
    for quest_cls in quest_cls_list:
        if quest_cls.question_name == question_name:
            return quest_cls
    return None
