
#@title Imports for agent building
import datetime
from typing import Sequence

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory, formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.components import sequential
from concordia.components.agent import (
  concat_act_component,
  person_representation,
  question_of_query_associated_memories,
  question_of_recent_memories,
)
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib


class ResponsibilityQuestion(question_of_recent_memories.QuestionOfRecentMemories):
  def __init__(
      self,
      agent_name:str,
      **kwargs,
  ):
    question = 'What responsibilities can {agent_name} share with others to improve cooperation and task completion?' #@param {"type":"string"}
    #@markdown The answer will have to start with this prefix
    answer_prefix = '{agent_name} can ' #@param {"type":"string"}
    #@markdown Flag that defines whether the answer will be added to memory
    add_to_memory = True # @param {"type":"boolean"}
    #@markdown If yes, the memory will start with this tag
    memory_tag = '[responsibility]' # @param {"type":"string"}
    question_with_name = question.format(agent_name=agent_name)
    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        components={},
        **kwargs,
    )


class TradeoffQuestion(question_of_recent_memories.QuestionOfRecentMemories):
  def __init__(
      self,
      agent_name:str,
      **kwargs,
  ):
    question = 'What trade-offs can {agent_name} propose to ensure that all agents feel their contributions are valued?' #@param {"type":"string"}
    #@markdown The answer will have to start with this prefix
    answer_prefix = '{agent_name} can ' #@param {"type":"string"}
    #@markdown Flag that defines whether the answer will be added to memory
    add_to_memory = True # @param {"type":"boolean"}
    #@markdown If yes, the memory will start with this tag
    memory_tag = '[trade off]' # @param {"type":"string"}
    question_with_name = question.format(agent_name=agent_name)
    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        components={},
        **kwargs,
    )


class MoralQuestion(question_of_recent_memories.QuestionOfRecentMemories):
  def __init__(
      self,
      agent_name:str,
      **kwargs,
  ):
    question = 'What potential harm could the group’s decisions cause, and how can {agent_name} help mitigate it?' #@param {"type":"string"}
    #@markdown The answer will have to start with this prefix
    answer_prefix = '{agent_name} can ' #@param {"type":"string"}
    #@markdown Flag that defines whether the answer will be added to memory
    add_to_memory = True # @param {"type":"boolean"}
    #@markdown If yes, the memory will start with this tag
    memory_tag = '[moral]' # @param {"type":"string"}
    question_with_name = question.format(agent_name=agent_name)
    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        components={},
        **kwargs,
    )

def _make_about_component(
    agent_name:str,
    measurements: measurements_lib.Measurements,
    model: language_model.LanguageModel,
    clock: game_clock.MultiIntervalClock,
):
  responsibility_question = ResponsibilityQuestion(
      agent_name=agent_name,
      model=model,
      logging_channel=measurements.get_channel('ResponsibilityQuestion').on_next,
  )
  tradeoff_question = TradeoffQuestion(
      agent_name=agent_name,
      model=model,
      logging_channel=measurements.get_channel('TradeoffQuestion').on_next,
  )
  moral_question = MoralQuestion(
      agent_name=agent_name,
      model=model,
      logging_channel=measurements.get_channel('MoralQuestion').on_next,
  )

  return (responsibility_question, tradeoff_question, moral_question)

class ActionQuestion(question_of_recent_memories.QuestionOfRecentMemories):
  def __init__(
      self,
      agent_name:str,
      **kwargs,
  ):
    question = 'Given the current circumstances and the information available, what action would {agent_name} take next to make all the member\'s feel better?' #@param {"type":"string"}
    #@markdown The answer will have to start with this prefix
    answer_prefix = '{agent_name} would ' #@param {"type":"string"}
    #@markdown Flag that defines whether the answer will be added to memory
    add_to_memory = True # @param {"type":"boolean"}
    #@markdown If yes, the memory will start with this tag
    memory_tag = '[action]' # @param {"type":"string"}
    question_with_name = question.format(agent_name=agent_name)
    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        components={
          'ResponsibilityQuestion': f'{agent_name}\'s responsibility:\n',
          'TradeoffQuestion': f'{agent_name}\'s trade-offs:\n',
          'MoralQuestion': f'{agent_name}\'s morals:\n',
          'Observation': '\nObservation',
          'ObservationSummary': '\nSummary of recent observations'
        },
        **kwargs,
    )

def _make_questions_component(
    agent_name:str,
    measurements: measurements_lib.Measurements,
    model: language_model.LanguageModel,
    clock: game_clock.MultiIntervalClock,
):
  acton_question = ActionQuestion(
      agent_name=agent_name,
      model=model,
      logging_channel=measurements.get_channel('ActionQuestion').on_next,
  )
  return (acton_question,)


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__

def build_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build an agent.

  Args:
    config: The agent config to use.
    model: The language model to use.
    memory: The agent's memory object.
    clock: The clock to use.
    update_time_interval: Agent calls update every time this interval passes.

  Returns:
    An agent.
  """
  del update_time_interval
  if not config.extras.get('main_character', False):
    raise ValueError(
        'This function is meant for a main character '
        'but it was called on a supporting character.'
    )

  agent_name = config.name

  raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)

  measurements = measurements_lib.Measurements()
  instructions = agent_components.instructions.Instructions(
      agent_name=agent_name,
      logging_channel=measurements.get_channel('Instructions').on_next,
  )

  time_display = agent_components.report_function.ReportFunction(
      function=clock.current_time_interval_str,
      pre_act_key='\nCurrent time',
      logging_channel=measurements.get_channel('TimeDisplay').on_next,
  )

  observation_label = '\nObservation'
  observation = agent_components.observation.Observation(
      clock_now=clock.now,
      timeframe=clock.get_step_size(),
      pre_act_key=observation_label,
      logging_channel=measurements.get_channel('Observation').on_next,
  )
  observation_summary_label = 'Summary of recent observations'
  observation_summary = agent_components.observation.ObservationSummary(
      model=model,
      clock_now=clock.now,
      timeframe_delta_from=datetime.timedelta(hours=4),
      timeframe_delta_until=datetime.timedelta(hours=0),
      pre_act_key=observation_summary_label,
      logging_channel=measurements.get_channel('ObservationSummary').on_next,
  )
  relevant_memories_label = '\nRecalled memories and observations'
  relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
      model=model,
      components={
          _get_class_name(observation_summary): observation_summary_label,
          _get_class_name(time_display): 'The current date/time is'},
      num_memories_to_retrieve=10,
      pre_act_key=relevant_memories_label,
      logging_channel=measurements.get_channel('AllSimilarMemories').on_next,
  )

  if config.goal:
    goal_label = '\nOverarching goal'
    overarching_goal = agent_components.constant.Constant(
        state=config.goal,
        pre_act_key=goal_label,
        logging_channel=measurements.get_channel(goal_label).on_next)
  else:
    goal_label = None
    overarching_goal = None

  person_representations = agent_components.person_representation.PersonRepresentation(
    model=model,
    additional_questions=[
      'Taking note of all the information above, what personality type will the character most likely get in 16 Personalities (MBTI) test?',
      'Based on the statements provided, what are {agent_name}\'s strengths and skills?',
      'What motivates the character?',
    ],
    logging_channel=measurements.get_channel('PersonRepresentation').on_next
  )

  about_component = _make_about_component(
      agent_name=agent_name,
      model=model,
      clock=clock,
      measurements=measurements
  )


  question_components = _make_questions_component(
      agent_name=agent_name,
      model=model,
      clock=clock,
      measurements=measurements
  )

  core_components = (
    instructions,
    time_display,
    observation,
    observation_summary,
    relevant_memories,
    person_representations,
    # sequential.Sequential(name='QuestionsComponent', components=about_component),
    # question_components
  )

  entity_components = core_components + tuple(about_component) + tuple(question_components)
  components_of_agent = {
      _get_class_name(component): component for component in entity_components
  }

  components_of_agent[
      agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME
  ] = agent_components.memory_component.MemoryComponent(raw_memory)
  component_order = list(components_of_agent.keys())
  if overarching_goal is not None:
    components_of_agent[goal_label] = overarching_goal
    # Place goal after the instructions.
    component_order.insert(1, goal_label)

  act_component = agent_components.concat_act_component.ConcatActComponent(
      model=model,
      clock=clock,
      component_order=component_order,
      logging_channel=measurements.get_channel('ActComponent').on_next,
  )

  agent = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=agent_name,
      act_component=act_component,
      context_components=components_of_agent,
      component_logging=measurements,
  )

  return agent
