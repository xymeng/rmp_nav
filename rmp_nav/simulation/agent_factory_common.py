all_agents = []
public_agents = []
private_agents = []
agents_dict = {}


def add_agent(a):
    all_agents.append(a)
    if a.__name__.startswith('_'):
        private_agents.append(a)
    else:
        public_agents.append(a)
    agents_dict[a.__name__] = lambda **kwargs: a(name=a.__name__, **kwargs)
    return a
