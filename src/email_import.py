"""Local EML extraction without network access or attachment execution."""
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
import re
MAX_BYTES=5*1024*1024
MAX_TEXT=50000
class VisibleHTML(HTMLParser):
    def __init__(self):super().__init__(convert_charrefs=True);self.parts=[];self.hidden=0
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style','head'):self.hidden+=1
        if not self.hidden and tag in ('p','div','br','li','tr','h1','h2','h3'):self.parts.append('\n')
    def handle_endtag(self,tag):
        if tag in ('script','style','head'):self.hidden=max(0,self.hidden-1)
        if not self.hidden and tag in ('p','div','li','tr'):self.parts.append('\n')
    def handle_data(self,data):
        if not self.hidden:self.parts.append(data)
def decode(part):
    data=part.get_payload(decode=True)
    if data is None:return str(part.get_payload()) if not part.is_multipart() else ''
    try:return data.decode(part.get_content_charset() or 'utf-8',errors='replace')
    except (LookupError,UnicodeError):return data.decode('utf-8',errors='replace')
def body(part):
    if part.get_content_disposition()=='attachment' or part.get_filename() or part.get_content_type()=='message/rfc822':return '',False
    if part.is_multipart():
        children=[body(x) for x in part.iter_parts()]
        if part.get_content_subtype()=='alternative':
            plain=[x for x in children if x[0].strip() and not x[1]]
            return plain[0] if plain else next((x for x in children if x[0].strip()),('',False))
        return '\n\n'.join(x[0] for x in children if x[0].strip()),any(x[1] for x in children if x[0].strip())
    kind=part.get_content_type()
    if kind=='text/plain':return decode(part),False
    if kind=='text/html':
        parser=VisibleHTML();parser.feed(decode(part));return ''.join(parser.parts),True
    return '',False
def extract_email(raw):
    if not raw:raise ValueError('The email file is empty.')
    if len(raw)>MAX_BYTES:raise ValueError('The email exceeds 5 MB.')
    try:message=BytesParser(policy=policy.default).parsebytes(raw)
    except Exception as e:raise ValueError('The email could not be parsed.') from e
    if not any(message.get(x) for x in ('Subject','From','To','Date','MIME-Version','Content-Type')):raise ValueError('No recognizable email headers were found. Export the message as an .eml file.')
    subject=re.sub(r'\s+',' ',str(message.get('Subject',''))).strip()
    text,html=body(message);text=re.sub(r'\n[ \t]*\n(?:[ \t]*\n)+','\n\n',text.replace('\r\n','\n')).strip()
    if not text:raise ValueError('No readable message body was found. Attachments and encrypted content are not supported.')
    combined=(subject+'\n\n'+text).strip()
    if len(combined)>MAX_TEXT:raise ValueError('Subject and body exceed 50,000 characters. Copy and edit the relevant text instead; no automatic truncation was applied.')
    return {'subject':subject,'body':text,'text':combined,'used_html':html,'warnings':['Attachments are ignored. Quoted replies and signatures may remain.','HTML was converted to text; verify the extracted content.' if html else 'Verify the subject and body before classification.']}
