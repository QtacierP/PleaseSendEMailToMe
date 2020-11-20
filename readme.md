# Please Send E-mail to me

---------------

It is a toy tool to help you get the experiment performance intermediately. Once the experiment has been finished, please send e-mail to me!

This is a demo:

```python
# Plase send E-mail to me
host_addr = 'xxx.smtp.com'
send_addr = 'xxx.gmail.com'
rec_addr = 'xxxx.gmail.com'
user = 'xxxx'
password = 'xxxxxxxxx'
sender = EMailSender(send_addr=send_addr, user=user, password=password, host=host_addr)
t = time.asctime(time.localtime(time.time()))
sender.send(subject='Information update at {}'.format(t),
            msg='Congratulations! We are pleased to inform you that your experiment has finished at {}, '
                'this is the final accuracy on the testset {} % \n'.format(t, acc),
            rec_addr=rec_addr)
```

 