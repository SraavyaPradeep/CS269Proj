# CS269Proj

Link to Final Presentation: https://docs.google.com/presentation/d/1EaxvL80_Jj75B19BbCwCM8yM8FCDj3Yuph0fGXSiHrU/edit#slide=id.gd431007ba2_0_215

## Workflow

<img width="1239" alt="Screenshot 2024-06-06 at 11 23 06 AM" src="https://github.com/SraavyaPradeep/CS269Proj/assets/46724697/b29df8b7-2c61-4833-a504-7e05d24d9685">

## Instruction

### Generating the Occupancy feature

```shell
cd ./OccNerf
bash new_runner.sh
```

if you want to generate the feature for val set, please modify the code in the line 165 of new_runner.py as follows.

```python
datasets_dict = {
            # "ddad": datasets.DDADDatasetRevision,
            "nusc": datasets.NuscDataset,
            # "kitti": datasets.KittiDataset,
        }
```