from s2s_vla.adapters.http import HTTPPolicy
from s2s_vla.adapters.example import ExampleEnvironment
from s2s_vla.interfaces import Runtime


def create_runtime(method, policy_url="http://127.0.0.1:3200", noise=0.015):
    return Runtime(ExampleEnvironment(method.max_steps, noise), HTTPPolicy(policy_url))
