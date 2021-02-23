from database.data_mappers.classification_group import viNetClassGroupMapper, viNetClassGroup
from database.data_mappers.vinet_frame_label import viNetTrainingFrameLabelMapper

# Step 1. Create classification group metadata
group_name = 'Vattenfall Proper v7'
comment = '6 class group without Other-Avian'
class_group = viNetClassGroup(group_name, comment)

# Step 2. Group classifications. i.e. Map classifications to a class group.
grouping = {
    # viNetCategory('Golden-Eagle', True): [viNetCategory('Golden-Eagle', True)],
    viNetTrainingFrameLabelMapper('Crow', False): [viNetTrainingFrameLabelMapper('Crow-sp.', False),
                                                   viNetTrainingFrameLabelMapper('Raven', False),
                                                   viNetTrainingFrameLabelMapper('Common-Raven', False)],
    viNetTrainingFrameLabelMapper('White-Tailed-Eagle', True): [viNetTrainingFrameLabelMapper('White-Tailed-Eagle', True)],
    viNetTrainingFrameLabelMapper('Gull', False): [viNetTrainingFrameLabelMapper('Gull', False),
                                                   viNetTrainingFrameLabelMapper('Herring-Gull', False),
                                                   viNetTrainingFrameLabelMapper('Great-Black-Backed-Gull', False),
                                                   viNetTrainingFrameLabelMapper('Common-Gull', False)],
    viNetTrainingFrameLabelMapper('Other-Avian-Gotland', False): [viNetTrainingFrameLabelMapper('Other-Avian-Gotlan-V2', False)],
    viNetTrainingFrameLabelMapper('Buzzard', False): [viNetTrainingFrameLabelMapper('Buzzard', False),
                                                      viNetTrainingFrameLabelMapper('Common-Buzzard', False),
                                                      viNetTrainingFrameLabelMapper('Hawk', False),
                                                      viNetTrainingFrameLabelMapper('Northern-Goshawk', False),
                                                      viNetTrainingFrameLabelMapper('Osprey', False),
                                                      viNetTrainingFrameLabelMapper('Harrier', False),
                                                      viNetTrainingFrameLabelMapper('Peregrine-Falcon', False)]
    }

for key, value in grouping.items():
    cg = viNetClassGroupMapper()
    cg.insert(class_group)
    cg.map_classifications(class_group, value, key)
