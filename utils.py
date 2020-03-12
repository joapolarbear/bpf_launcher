import argparse
import os
import json

parser = argparse.ArgumentParser(description="Trace Analysis",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("-s", action="store_true", help="sort the output result")
parser.add_argument("--option", type=str, default="status",
					help="")

parser.add_argument("--retstr", type=str, help="The returned string")
parser.add_argument("--target", type=str, default=None, help="The name of the target container")
parser.add_argument("--command", type=str, help="Command")
parser.add_argument("--bash_arg", type=str, help="args of the bash script")

args = parser.parse_args()

def list2bash_array(l):
	return " ".join(l)

if args.option == "status":
	ip_add = args.retstr.split("spawn")[1].split('\n')[0]
	_retstr = "\n".join(args.retstr.split("root@")[-2].split("\n")[1:])
	if args.target is None:
		print("#" * 50 + ip_add)
	if not (args.target is not None and args.target not in _retstr):
		# print("container " + args.target + " is still running or has not been removed.")
		print(_retstr)
	else:
		print("container " + args.target + " has been removed.")
elif args.option == "gpu" or args.option == "ip" or args.option == "tc":
	_bash_arg = args.bash_arg.split(",")
	assert len(_bash_arg) >= 3
	assert _bash_arg[2].isdigit()

elif "readcfg_" in args.option:
	with open("./cfg.json", "r") as fp:
		cfg = json.load(fp)
	target = args.option.split("readcfg_")[1]
	print(list2bash_array(cfg[target])) 

