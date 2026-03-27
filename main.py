import os
import time
import ast
import inspect
import yaml
import argparse
import shutil

from src.agents.tdlambda import *
from src.agents.pearl import PEARL
from src.misc.log import LogExperiments
from src.abstraction.abstraction import Abstraction
from src.misc import env_builder


argparser = argparse.ArgumentParser()
argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--method', type=str, default="PEARL")
argparser.add_argument('--partitioning', type=str, default="flexible")
argparser.add_argument('--trial', type=int, default=1)
argparser.add_argument('--domain', type=str, default="office")
argparser.add_argument('--config', type=int, default=0)
argparser.add_argument('--reuse_cat_file', type=str, default=None, required=False)
argparser.add_argument('--gamma', type=float, default=0.9)  
argparser.add_argument('--alpha', type=float, default=0.1)
argparser.add_argument('--epsilon_min', type=float, default=0.1)
argparser.add_argument('--decay', type=float, default=0.9)
argparser.add_argument('--lamda', type=float, default=0.9)
argparser.add_argument('--init_beta', type=float, default=1.0)
argparser.add_argument('--decay_beta_amount', type=float, default=0.05)
argparser.add_argument('--min_beta', type=float, default=0.15)
argparser.add_argument('--k_cap', type=int, default=10)
argparser.add_argument('--k_cap_actions', type=int, default=10)
argparser.add_argument('--bootstrap', type=str, default="from_estimated_concrete")
argparser.add_argument('--refinement_method', type=str, default="deliberative")   
argparser.add_argument('--init_state_abs_level', type=int, default=0)
argparser.add_argument('--init_action_abs_level', type=int, default=1)
argparser.add_argument('--max_clusters', type=int, default=10)
argparser.add_argument('--min_samples', type=int, default=2)
argparser.add_argument('--kernel', type=str, default="linear")
argparser.add_argument('--allowed_diff_to_refine', type=float, default=0.001)
argparser.add_argument('--plot_abstractions', action="store_true")
argparser.add_argument('--episode_max', type=int, default=1000)
argparser.add_argument('--step_max', type=int, default=1000)
argparser.add_argument('--abs_interval', type=int, default=100)
argparser.add_argument('--eval_episodes', type=int, default=100)
argparser.add_argument('--interactive', action="store_true")
argparser.add_argument("--yaml", type=str, default=None)
argparser.add_argument("--maximum_state_variables_to_split", type=str, default=None)
argparser.add_argument("--result_dir", type=str, default="None")
argparser.add_argument("--method_dir", type=str, default="None")

## some specific params
argparser.add_argument("--map_name", type=str, default="office_harder.cfg")
argparser.add_argument("--start_pos", type=ast.literal_eval, default="(0.05, 0.05)")
argparser.add_argument("--target_pos", type=ast.literal_eval, default="(0.95, 0.95)")

## office params
argparser.add_argument("--coffee_pos", type=ast.literal_eval, default="(0.1, 0.8)")
argparser.add_argument("--mail_pos", type=ast.literal_eval, default="(0.54, 0.55)")

## multicity params
argparser.add_argument("--package_pos", type=ast.literal_eval, default="(0.2, 0.7)")
argparser.add_argument("--package_city", type=ast.literal_eval, default="1")
argparser.add_argument("--airport_city1", type=ast.literal_eval, default="[(0.35, 0.7), (0.9, 0.35), (0.9,0.7)]")
argparser.add_argument("--airport_city2", type=ast.literal_eval, default="[(0.2,0.25), (0.5,0.7), (0.7,0.25) ]")
argparser.add_argument("--airport_city3", type=ast.literal_eval, default="[(0.4, 0.7), (0.7, 0.35), (0.75,0.7)]")
argparser.add_argument("--agent_city", type=ast.literal_eval, default="0")
argparser.add_argument("--target_city", type=ast.literal_eval, default="2")


def load_yaml(yaml_file): 
    with open(yaml_file, "r") as f:
        yaml_args = yaml.safe_load(f)

# Convert to list of strings like ['--learning_rate', '0.001', ...]
    arg_list = [str(k) + '=' + str(v) if '=' not in k else f"{k}" for k, v in yaml_args.items()]
    arg_list = [item for pair in [a.split('=') for a in arg_list] for item in pair]  # flatten
    return arg_list

args = argparser.parse_args()

if args.yaml is not None: 
    yaml_args_list = load_yaml(args.yaml)
    yaml_args = argparser.parse_args(yaml_args_list)
    args = argparser.parse_args(namespace=yaml_args)

# base params
method = args.method
partitioning = args.partitioning
trial = args.trial
domain = args.domain
reuse_cat_file = args.reuse_cat_file
config = args.config
result_dir = args.result_dir
method_dir = args.method_dir

# TD(lambda) hyperparameters
gamma = float(args.gamma)
alpha = float(args.alpha)
epsilon_min = float(args.epsilon_min)
decay = float(args.decay)
_lambda = float(args.lamda)
init_beta = float(args.init_beta)
decay_beta_amount = float(args.decay_beta_amount)
min_beta = float(args.min_beta)


# PEARL parameters
k_cap = int(args.k_cap)
k_cap_actions = int(args.k_cap_actions)
bootstrap = args.bootstrap
refinement_method = args.refinement_method
init_state_abs_level = int(args.init_state_abs_level)
init_action_abs_level = int(args.init_action_abs_level)
flexible_refinement = True if args.partitioning=="flexible" else False
max_clusters = int(args.max_clusters)
min_samples = int(args.min_samples)
kernel = args.kernel
allowed_diff_to_refine = float(args.allowed_diff_to_refine)
plot_abstractions = bool(args.plot_abstractions)

#PEARL parameters
episode_max = int(args.episode_max)
step_max = int(args.step_max)
abs_interval = int(args.abs_interval)
eval_episodes = int(args.eval_episodes)

args_dict = vars(args)

env_alg_to_time = dict()
if trial is None:
    trials = [1,2,3,4,5,6,7,8,9,10]
else:
    trials = [trial]

for trial in trials:
    basepath = os.getcwd()
    directory = f"{basepath}/{result_dir}/{domain}/{method_dir}/trial_{str(trial)}"
    print(f"Running trial {trial} in directory {directory}")
    if not os.path.exists(directory):
        os.makedirs(directory)

    yaml_source = f"{basepath}/yamls/{domain}_{partitioning}_{config}.yaml"
    yaml_destination = f"{directory}/{domain}_{partitioning}_{config}.yaml"
    if os.path.exists(yaml_source):
        shutil.copy(yaml_source, yaml_destination)
    else:
        print(f"Warning: {yaml_source} does not exist.")

    if reuse_cat_file is not None:
        reuse_cat_path = f"{directory}/{reuse_cat_file}"
    else:
        reuse_cat_path = None
    start_time = time.time()

    #_______________________________________________________________________________________

    builder_func = env_builder.get_env_builder(domain).__init__
    sig = inspect.signature(builder_func)
    filtered_args = {k:v for k,v in args_dict.items() if k in sig.parameters}
    env = env_builder.get_env_builder(domain)(**filtered_args)

    sig = inspect.signature(env.initialize_problem)
    filtered_args = {k:v for k,v in args_dict.items() if k in sig.parameters}
    env.initialize_problem(**filtered_args)

    seed = trial
    env.seed(seed)

    agent = AbstractTDlambdaAgent(seed = seed,
                                is_action_space_discrete = env.is_action_space_discrete,
                                action_size = env.action_size,
                                gamma = gamma, 
                                alpha = alpha, 
                                eps_min = epsilon_min, 
                                decay = decay,
                                _lambda = _lambda,
                                epsilon = 1)

    agent_con = None
    abstract = Abstraction(seed = seed,
                            env = env, 
                            agent = agent, 
                            agent_con = agent_con, 
                            k_cap = k_cap,
                            k_cap_actions = k_cap_actions,
                            bootstrap = bootstrap,
                            refinement_method = refinement_method,
                            init_action_abs_level = init_action_abs_level,
                            init_state_abs_level = init_state_abs_level,
                            flexible_refinement = flexible_refinement,
                            max_clusters = max_clusters,
                            min_samples = min_samples,
                            kernel = kernel,
                            allowed_diff_to_refine = allowed_diff_to_refine,
                            directory = directory,
                            reuse_cat_path = reuse_cat_path,
                            plot_abstractions = plot_abstractions,
                            init_beta = init_beta, 
                            decay_beta_amount = decay_beta_amount, 
                            min_beta = min_beta
                            )
    agent._abstract = abstract
    log = LogExperiments(directory = directory)

    pearl = PEARL(seed = seed,
                env = env, 
                agent = agent, 
                agent_con = agent_con, 
                abstract = abstract, 
                log = log, 
                episode_max = episode_max, 
                step_max = step_max, 
                abs_interval = abs_interval, 
                eval_episodes = eval_episodes, 
                directory = directory)
    eval_score = pearl.main()

    env_alg_to_time[trial] = float(round((time.time() - start_time),2))
    for trial,t in env_alg_to_time.items():
        print(f"{trial} time: {str(t)} s")
    print(f"Evaluation Score: {eval_score}")