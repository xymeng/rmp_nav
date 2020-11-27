from __future__ import print_function
from . import agent_factory_common

'''
Unless absolutely necessary, we should only add agents to this file , but not modify any existing 
agent.

Agents that start with '_' are private agents. For example, agents that are used for training.
'''

all_agents = agent_factory_common.all_agents
public_agents = agent_factory_common.public_agents
private_agents = agent_factory_common.private_agents
agents_dict = agent_factory_common.agents_dict


# Import agents
from . import agent_factory_rccar, agent_factory_minirccar, agent_factory_turtlebot
