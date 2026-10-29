# Hather: Saudi scenario list for the teacher

The teacher (Qwen 3.8 27B) turns each scenario into realistic Arabic messages
of the kind people in Saudi Arabia actually receive. The scenarios are written
in English on purpose: the teacher writes the Arabic.

Format (one per line, parsed by `make_runA_jobs.py`, keep the `|` separators):
`- ID | label | channel | sender / brand | what the message does`

Edit freely: add, remove or reword lines. Keep about half scam and half legit.

## Scam

- S01 | scam | sms | Absher (أبشر) | Says the user's Absher account is suspended or will be closed and asks them to update their data through a link within 24 hours.
- S02 | scam | sms | Saudi Post / SPL (سبل) | Says a parcel is held at the warehouse and asks for a small customs or redelivery fee (5-30 SAR) through a payment link.
- S03 | scam | sms | Aramex / SMSA | Says a delivery attempt failed and asks the user to reschedule and pay a small fee through a link.
- S04 | scam | sms | A Saudi bank (Al Rajhi, SNB, Riyad Bank, Alinma) | Says the bank card has been blocked or suspended for a security reason and asks the user to reactivate it through a link.
- S05 | scam | whatsapp | Someone claiming to be a bank employee | Says there is a problem with the account and asks the user to share the verification code (OTP) they are about to receive.
- S06 | scam | sms | Traffic police / Muroor (المرور), Saher (ساهر) | Says there is an unpaid traffic fine with a limited-time discount and asks for payment through a link.
- S07 | scam | sms | STC / Mobily / Zain | Says the user won a prize or loyalty points and must pay a small fee or enter card details to claim it.
- S08 | scam | whatsapp | A recruiter from an unknown number | Offers easy remote work (liking videos, rating products) with daily pay, then asks for a deposit or a "task" payment.
- S09 | scam | sms | Najm (نجم) | Says there is an accident compensation or insurance claim waiting and asks the user to enter bank card details to receive it.
- S10 | scam | sms | ZATCA (هيئة الزكاة والضريبة والجمارك) | Says the user is eligible for a tax or VAT refund and asks for card details through a link.
- S11 | scam | sms | Citizen Account (حساب المواطن) | Says the user's support payment is stopped until they update their eligibility data through a link.
- S12 | scam | whatsapp | A travel agent from an unknown number | Offers a very cheap Umrah or Hajj package or a Nusuk permit, asks for a transfer to a personal account to book.
- S13 | scam | whatsapp | An "investment advisor" | Promises guaranteed daily profits from stocks, Tadawul tips or crypto, asks the user to transfer money to start.
- S14 | scam | sms | SADAD (سداد) | Says a bill is overdue and the service will be cut today unless paid through a link.
- S15 | scam | sms | Saudi Electricity Company (الكهرباء) | Says the user has an electricity bill refund or the power will be disconnected, asks for card details through a link.
- S16 | scam | whatsapp | A "family member" with a new number | Says "this is my new number", then asks for an urgent money transfer because of an emergency.
- S17 | scam | whatsapp | A stranger | Says they transferred money to the user by mistake, sends a fake receipt, and asks the user to send it back.
- S18 | scam | sms | Jawazat (الجوازات) | Tells a resident their Iqama or visa has a problem or fine and must be settled through a link to avoid deportation.
- S19 | scam | sms | Noon / Amazon.sa | Says an order has a payment problem or the user won a gift card, asks them to confirm card details through a link.
- S20 | scam | whatsapp | A friend's hacked account | Asks the user to forward the 6-digit WhatsApp code they just received "by mistake".

## Legit

- L01 | legit | sms | A Saudi bank | A one-time password (OTP) for a login or purchase, with "do not share this code with anyone". No link.
- L02 | legit | sms | A Saudi bank | A card purchase alert: amount in SAR, merchant name, last 4 card digits, remaining balance.
- L03 | legit | sms | A Saudi bank | Salary or transfer received, with amount and balance.
- L04 | legit | sms | Absher (أبشر) | A real reminder that a passport, ID or Iqama expires soon and can be renewed through the Absher app. No payment link.
- L05 | legit | sms | Saudi Post / SPL (سبل) | A real delivery update with a tracking number and the official site splonline.com.sa or the app.
- L06 | legit | sms | Aramex / SMSA | The parcel is out for delivery today, with the courier's arrival window.
- L07 | legit | sms | A clinic or hospital | An appointment reminder with date and time, and a number to call to cancel.
- L08 | legit | sms | Traffic police / Muroor (المرور) | A real notice that a traffic violation was recorded and can be viewed in Absher. No payment link.
- L09 | legit | sms | STC / Mobily / Zain | The monthly bill is issued, with the amount and "pay through the app".
- L10 | legit | sms | Nafath (نفاذ) | A login verification request asking the user to choose a number in the Nafath app.
- L11 | legit | whatsapp | A family member | Family chat: dinner plans, who is bringing what, what time to come.
- L12 | legit | whatsapp | A work colleague | The meeting moved to another time, or a request to review a file.
- L13 | legit | whatsapp | A friend | Plans for the weekend, a football match, or going to a café.
- L14 | legit | sms | Jahez / HungerStation / Keeta | The food order is on the way, with the estimated arrival time.
- L15 | legit | sms | A known retailer (Jarir, Extra, Centrepoint) | A real promotion with the official website and an opt-out code. No request for personal data.
- L16 | legit | sms | SADAD (سداد) | A new bill notice with biller number and amount, paid through the user's own banking app. No link.
- L17 | legit | sms | A university or school | A notice about exams, registration dates or a holiday.
- L18 | legit | sms | Careem / Uber | A ride receipt with fare and route.
- L19 | legit | sms | Sehhaty (صحتي) | A real vaccination or health appointment reminder in the Sehhaty app.
- L20 | legit | sms | Ehsan (إحسان) | A donation receipt thanking the user, with amount and reference number.
