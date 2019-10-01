################################################################################
# IMPORTANT NOTE !
# Because of the too long running time with our data augmentation, the models
# are not trained with the data augmentation here. In consequence there will
# certainly be some inconsitencies with the report!
#
# Running time on an intel i7 2.8 GHz : 740 seconds ≈ 12 minutes
################################################################################

import SimpleCNN as simpleCNN
import WeightSharing as WS
import GlobalPlot as plot
import Auxiliaryloss as AL

# %% run the Simple model
simpleCNN.run_model()

# %% run the Weight Sharing model
WS.run_model()

# %% run the Auxiliary Loss model
AL.run_model()

# %% plot the results
plot.plotResults()
