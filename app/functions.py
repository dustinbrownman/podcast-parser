import modal


def download_whisper():
    # Load the Whisper model
    import os
    import whisper
    print("Download the Whisper model")

    # Perform download only once and save to Container storage
    whisper._download(whisper._MODELS["medium"], '/content/podcast/', False)


stub = modal.Stub("corise-podcast-project")
corise_image = modal.Image.debian_slim().pip_install("feedparser",
                                                     "https://github.com/openai/whisper/archive/9f70a352f9f8630ab3aa0d06af5cb9532bd8c21d.tar.gz",
                                                     "requests",
                                                     "ffmpeg",
                                                     "openai",
                                                     "tiktoken",
                                                     "wikipedia",
                                                     "ffmpeg-python").apt_install("ffmpeg").run_function(download_whisper)


@stub.function(image=corise_image, timeout=30)
def get_podcast_data(rss_url, episode_index=0):
    import feedparser

    podcast_data = feedparser.parse(rss_url)
    feed = podcast_data['feed']
    episode = podcast_data.entries[episode_index]

    output = {
        "title": feed.get("title", ""),
        "image": feed.get("image").href,
        "episode": {
            "title": episode.get("title", ""),
            "description": episode.get("description", ""),
            "published": episode.get("published", ""),
            "author": episode.get("author", ""),
        }
    }

    for link in episode.get("links", []):
        if link['type'] == 'audio/mpeg':
            output['episode']['url'] = link.href

    return output


@stub.function(image=corise_image, gpu="any", timeout=600)
def get_transcribe_podcast(episode_url, local_path="/content/podcast/"):
    print("Starting Podcast Transcription Function")
    print("RSS URL read and episode URL: ", episode_url)

    # Download the podcast episode by parsing the RSS feed
    episode_name = "episode.mp3"

    from pathlib import Path
    p = Path(local_path)
    p.mkdir(exist_ok=True)

    print("Downloading the podcast episode")
    import requests
    with requests.get(episode_url, stream=True) as r:
        r.raise_for_status()
        episode_path = p.joinpath(episode_name)
        with open(episode_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print("Podcast Episode downloaded")

    # Load the Whisper model
    import os
    import whisper

    # Load model from saved location
    print("Load the Whisper model")
    model = whisper.load_model(
        'medium', device='cuda', download_root='/content/podcast/')

    # Perform the transcription
    print("Starting podcast transcription")
    result = model.transcribe(local_path + episode_name)

    # Return the transcribed text
    print("Podcast transcription completed, returning results...")

    return result['text']


@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_podcast_summary(transcript, description):
    import openai
    summary_prompt = f"""
    You are a personal assistant for a person who likes to listen to podcasts. Your job is to
    summarize the podcast transcript below. You will be graded on the following criteria:
    1. The summary you write is concise and captures the main points of the podcast
    2. The summary you write is not too long (1-2 paragraphs)
    3. The summary you write is grammatically correct

    The podcast has its own description of the podcast episode, which you can use as a starting
    point for your summary.

    Episode Description: "{description}"

    Episode Transcript: "{transcript}"
  """

    chatOutput = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": summary_prompt}
        ]
    )

    summary = chatOutput.choices[0].message.content

    return summary


@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_podcast_guest(transcript, author_data):
    import openai
    import wikipedia
    import json

    guests_prompt = f"""
    You are a personal assistant for a person who likes to listen to podcasts. Your job is to
    find interesting people who are speaking in the podcast transcript below. You will be graded
    on the following criteria:

    1. The people you find are speaking in the podcast
    2. You correct for any spelling mistakes in the names of the people you find, using the authors
    of the podcast as a reference: {author_data}

    Episode Transcript: "{transcript}"
    """

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[{"role": "user", "content": guests_prompt}],
        functions=[
            {
                "name": "get_interesting_people",
                "description": "Get information on the podcast guests using their full name and the name of the organization they are part of to search for them on Wikipedia",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "guests": {
                            "type": "array",
                            "description": "The list of guests who are speaking in the podcast",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "The full name of the guest who is speaking in the podcast",
                                    },
                                    "organization": {
                                        "type": "string",
                                        "description": "The full name of the organization that the podcast guest belongs to or runs",
                                    },
                                    "title": {
                                        "type": "string",
                                        "description": "The title, designation or role of the podcast guest in their organization",
                                    },
                                },
                                "required": ["name"],
                            },
                        },
                    },
                    "required": ["guests"],
                },
            }
        ],
        function_call={"name": "get_interesting_people"}
    )

    guests = []
    function_name = None
    response_message = completion["choices"][0]["message"]

    # The response indicates that the model wants to call the function
    if response_message.get("function_call"):
        function_name = response_message["function_call"]["name"]
        function_args = json.loads(
            response_message["function_call"]["arguments"])
        guests = function_args.get("guests")

    if function_name != "get_interesting_people":
        return []

    for guest in guests:
        wiki = None
        try:
            wiki = wikipedia.page(
                f"{guest.get('name')} {guest.get('title')} {guest.get('organization')}", auto_suggest=False)
        except:
            None

        if wiki is not None:
            guest.update({"wiki": {
                "title": wiki.title,
                "url": wiki.url,
                "summary": wiki.summary,
            }})
        else:
            print(f"No wikipedia page found for {guest.get('name')}")

    return guests


@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_podcast_highlights(podcast_transcript):
    import openai
    highlights_prompt = """
        You are a copywriter for a podcast company. Your job is to pick 2-3 highlights from the podcast
        transcript below. You will be graded on the following criteria:
        1. The highlights you pick are interesting and engaging
        2. The highlights you pick are from different parts of the podcast
        3. The highlights you pick are not too long (1-2 sentences)
        The highlights you pick should be returned in Markdown format. For example:
        - **Title of first highlight:** This is the first highlight I picked
        - **Title of second highlight:** This is the second highlight I picked
        - **Title of third highlight:** This is the third highlight I picked

    """

    request = highlights_prompt + podcast_transcript

    chat_output = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": request}
        ]
    )

    podcast_highlights = chat_output.choices[0].message.content

    return podcast_highlights


@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"), timeout=1200)
def process_podcast(url, episode_index=0, path="/content/podcast/"):
    podcast_data = get_podcast_data.remote(url, episode_index)

    if not podcast_data.get('episode', False) or not podcast_data['episode'].get('url', False):
        print("No podcast data found")
        return

    transcript = get_transcribe_podcast.remote(
        podcast_data['episode']['url'], path)

    output = {
        "podcast_details": podcast_data,
        "summary": get_podcast_summary.remote(transcript, podcast_data['episode']['description']),
        "guests": get_podcast_guest.remote(transcript, podcast_data['episode']['author']),
        "highlights": get_podcast_highlights.remote(transcript)
    }
    podcast_data['episode']['transcript'] = transcript

    return output


@stub.local_entrypoint()
def test_method(url):
    print("Test method called")
    result = process_podcast.remote(url)
    print(result)
    return result
