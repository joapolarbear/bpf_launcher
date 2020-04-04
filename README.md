# A Launcher for Distribute ML tasks with BytePS/Horovod

## Configuration
Before we start a trail, we need to specify some configuration information in `cfg.json`. We can add some configuration information to file `cfg.json`, which is organized in a JSON format. An example of the configuration file `cfg.json`, here the `host` field corresponds to a list of IP address of some host machines, and the field `visible_device` specify the visible devices on each machines.
```
{
	"host": [
		"xxx.xxx.xxx.xxx",
		"xxx.xxx.xxx.xxx"
		],
	"visible_device": 0,1
}
```
When we want to read configuration information about "host", we can enter the bash command `python3 utils.py --option readcfg_host`.

## Usage
Use the following command.
```bash
./start <option>
```

### Lauch a Trail
`--option start`: start remote tasks.

### Stop a Trail
`--option stop`:

### Check the Status
`--option status`: list the status of all containers.

### Collect Traces
`--option collect`: collect all traces to local.


