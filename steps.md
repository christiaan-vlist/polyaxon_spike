### Start-up

Go to dir

```bash
cd /Users/christian/Documents/Repositories/polyaxon_spike/polyaxon_code
```

### Train a model

Execute train job

```bash
polyaxon run -f train/polyaxonfile.yaml -l -p ml-serving
```

Note the uuid, we will need this for the next step. 

Check training progress in the CLI or the GUI: http://localhost:8000/ui/default/ml-serving/jobs. Can also do model comparisons in GUI.

Training a model saves it in ./model.joblib for now

(alternatively use 6dbb7848d57441528410e03143d7b3c9 as uuid)

### Serve the model

Start up server

```bash
polyaxon run -f flask_serving/polyaxonfile.yaml -p ml-serving -P uuid=<UUID FROM PREV STEP>
```

Again, note the uuid 

Check if server is doing well. Can use the GUI: http://localhost:8000/ui/default/ml-serving/services. Or CLI:

```bash
polyaxon ops service --external --url -p ml-serving -uid <UUID FROM PREV STEP> 
```

(alternatively use 2bb891ae232a4501b643a7c2758b2c0d as service uuid: 

```bash
polyaxon run -f flask_serving/polyaxonfile.yaml -p ml-serving -P uuid=6dbb7848d57441528410e03143d7b3c9
```

```bash
polyaxon ops service --external --url -p ml-serving -uid 2bb891ae232a4501b643a7c2758b2c0d
```
)

### Do some wonderful predictions

```
curl localhost:8000/rewrite-services/v1/polyaxon/default/ml-serving/runs/<UUID OF SERVICE>/api/v1/predict --request POST \
    --header "Content-Type: application/json" \
    --data '{"from_ids": "test_christiaan", "n":"5"}'
```

Ready made:

```
curl localhost:8000/rewrite-services/v1/polyaxon/default/ml-serving/runs/2bb891ae232a4501b643a7c2758b2c0d/api/v1/predict --request POST \
    --header "Content-Type: application/json" \
    --data '{"from_ids": "test_christiaan", "n":"5"}'
```