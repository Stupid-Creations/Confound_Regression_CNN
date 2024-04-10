from Confound_Regression import *

address = 'address'
directory = os.listdir(address)

def make_scanset(scans,address = 'address'):
    inputs = []
    for i in range(113):
      if len(directory[i].split('.')) < 2:
          #this is particular to the dataset used for this model
        inputs.append(read_scan(address+directory[i]+"/RAW/"+directory[i]+"_mpr-1_anon.hdr"))
        return inputs

#Gets the labels and confounder information from a CSV file
def labels_and_confounds(csvaddress'address'):
    labels = []
    clabels = []
    wanted = ["M/F","Age","Educ"]

    with open(csvaddress) as file:
      reader = csv.reader(file)
      i = 0
      for row in reader:
        labels.append(row[7])
        clabels.append([row[1],row[3],row[4]])
        i += 1
        if i == len(inputs):
          break
    return labels,clabels

labels,clabels = labels_and_confounds()
inputs = make_scanset()

model = triple_adverse()

model.train(clabels,inputs,labels,1)

tlabel,tclabel = labels_and_confounds(csvaddress = "testaddresscsv")
tinputs = labels_and_confounds(address = "testaddress")

totalset = tf.Dataset.from_tensor_slices(tinputs,tlabel)

loss,accuracy = model.model.evaluate(totalset)

confound_regressor_MSE = self.test_regressor(tclabel,tinputs)
