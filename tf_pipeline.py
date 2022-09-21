import tensorflow as tf

monthly_sales = [41, 42, -108, 32, 55, 67, -1, 34]

tf_dataset = tf.data.Dataset.from_tensor_slices(monthly_sales)

print(tf_dataset)

for sales in tf_dataset.as_numpy_iterator():
    print(sales)

# print only first 3 elements:
for sales in tf_dataset.take(3):
    print(sales.numpy())

# filtering data:
tf_filtered = tf_dataset.filter(lambda x: x>0)
print("Filter ")
for sales in tf_filtered.as_numpy_iterator():
    print(sales)

# multiply all elements with 3:
print("Multiply all elements by 3")
tf_multiplied = tf_dataset.map(lambda x: x*3)
for sales in tf_multiplied.as_numpy_iterator():
    print(sales)

# shuffle the elements:
print("Shuffle the dataset")
tf_dataset = tf_dataset.shuffle(2)
for sales in tf_dataset.as_numpy_iterator():
    print(sales)

# create batches:
print("Create batches of 2 elements:")
for sales in tf_dataset.batch(2):
    print(sales.numpy())




