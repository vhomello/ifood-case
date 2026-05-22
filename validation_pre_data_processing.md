## Offers

1. The count is right? We dont have the original data count
2. The schema is right? Yes
3. All BOGO discount = min_value?
4. All Discount discount < min_value?
5. Transaction table has all the offers id?
6. All offers id in transactions?

## Profile
1. The count is right? Yes
2. The schema is right? No, we need to fix registered_on column to read YYYYMMDD #DONE
2. Age 118 is null? yes, when all others columns is null the age is 118

## Transactions
1. offer id = offer_id? yes, i used a coalesce
2. account e offer estão nesse dataset?
3. valores reawrd são iguais e menores que o discount?
4. quantas offer tem uma account
5. quantas transações tem uma account
6. monitoramos uma account por quanto tempo nesse dataset?


