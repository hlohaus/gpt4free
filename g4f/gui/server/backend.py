import g4f

from flask      import request
from .internet  import get_search_message

g4f.debug.logging = True

class Backend_Api:
    def __init__(self, app) -> None:
        self.app = app
        self.routes = {
            '/backend-api/v2/models': {
                'function': self.models,
                'methods' : ['GET']
            },
            '/backend-api/v2/providers': {
                'function': self.providers,
                'methods' : ['GET']
            },
            '/backend-api/v2/conversation': {
                'function': self._conversation,
                'methods': ['POST']
            },
            '/backend-api/v2/gen.set.summarize:title': {
                'function': self._gen_title,
                'methods': ['POST']
            },
            '/backend-api/v2/error': {
                'function': self.error,
                'methods': ['POST']
            }
        }
    
    def error(self):
        print(request.json)
        
        return 'ok', 200
    
    def models(self):
        return g4f._all_models
    
    def providers(self):
        return [
            provider.__name__ for provider in g4f.Provider.__providers__
            if provider.working and provider is not g4f.Provider.RetryProvider
        ]
    
    def _gen_title(self):
        return {
            'title': ''
        }
    
    def _conversation(self):
        try:
            #jailbreak = request.json['jailbreak']
            web_search = request.json['meta']['content']['internet_access']
            messages = request.json['meta']['content']['parts']
            if web_search:
                messages[-1]["content"] = get_search_message(messages[-1]["content"])
                print(messages[-1]["content"])
            model = request.json.get('model')
            model = model if model else g4f.models.default
            provider = request.json.get('provider', 'Auto').replace('g4f.Provider.', '')
            provider = provider if provider != "Auto" else None
            if provider != None:
                provider = g4f.Provider.ProviderUtils.convert.get(provider)

            response = g4f.ChatCompletion.create(
                model=model,
                provider=provider,
                messages=messages,
                stream=True,
                ignore_stream_and_auth=True
            )

            return self.app.response_class(response, mimetype='text/event-stream')

        except Exception as e:
            print(e)
            return {
                'code'   : 'G4F_ERROR',
                '_action': '_ask',
                'success': False,
                'error'  : f'an error occurred {str(e)}'}, 400
