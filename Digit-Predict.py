from digit_rec import predict, load

model = load()

#Instead of test.csv you can insert your testable file to predict the digits from the csv type file.
print("Please wait your file is being prepared.")
print(predict(model, "test.csv"))
print("Your file is ready at the outside")