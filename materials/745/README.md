# 745 外部类型合同与固定读者

writer-contract.json是实际外部写者声明；reader-contract.json为固定客户合同。input.json给两行、两次不安全合同更正及两个小兼容例，没有兼容编译器。default只在整个writer缺列时补，不给现有null填默认。

目标输出reader-records/v1，含reader_version、rows，每行有唯一row_id、values和sources；values按field_id给全部reader列，数值为最简分数字符串、TEXT为字符串、null为JSON null。sources记录原field_id或default依据。reader-example-records.json是手写外部格式样例。

reader_client.py CONTRACT.json RECORDS.json真正读取转换文件核数值域、null及列集合，输出数值列精确合计。它不判断writer全域兼容，不生成适配器。初例amount总-1、status均ok。INT8全域不在DECIMAL(2,0)内，INT16全域不在DECIMAL(3,0)内；样本1/-2虽能装入也不能据此通过。DECIMAL(3,1)→(4,2)保值，nullable→非nullable不安全。
