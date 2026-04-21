ef calculator():
    while True:
        print("\nВыберите операцию:")
        print("1. Сложение")
        print("2. Вычитание")
        print("3. Умножение")
        print("4. Деление")
        print("5. Выход")

        try:
            choice = int(input("Введите номер операции: "))
        except ValueError:
            print("Некорректный ввод. Пожалуйста, введите число от 1 до 5.")
            continue

        if choice == 5:
            print("Выход из калькулятора.")
            break

        try:
            num1 = float(input("Введите первое число: "))
            num2 = float(input("Введите второе число: "))
        except ValueError:
            print("Некорректный ввод. Пожалуйста, введите числа.")
            continue

        if choice == 1:
            result = num1 + num2
            print(f"{num1} + {num2} = {result}")
        elif choice == 2:
            result = num1 - num2
            print(f"{num1} - {num2} = {result}")
        elif choice == 3:
            result = num1 * num2
            print(f"{num1} * {num2} = {result}")
        elif choice == 4:
            if num2 == 0:
                print("Ошибка! Деление на ноль невозможно.")
            else:
                result = num1 / num2
                print(f"{num1} / {num2} = {result}")
        else:
            print("Некорректный выбор операции.")

if __name__ == "__main__":
    calculator()