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


@stub.function(image=corise_image, gpu="any", timeout=600)
def get_transcribe_podcast(rss_url, local_path):
    print("Starting Podcast Transcription Function")
    print("Feed URL: ", rss_url)
    print("Local Path:", local_path)

    # Read from the RSS Feed URL
    import feedparser
    intelligence_feed = feedparser.parse(rss_url)
    podcast_title = intelligence_feed['feed']['title']
    episode_title = intelligence_feed.entries[0]['title']
    episode_image = intelligence_feed['feed']['image'].href
    for item in intelligence_feed.entries[0].links:
        if (item['type'] == 'audio/mpeg'):
            episode_url = item.href
    episode_name = "podcast_episode.mp3"
    print("RSS URL read and episode URL: ", episode_url)

    # Download the podcast episode by parsing the RSS feed
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
    output = {}
    output['podcast_title'] = podcast_title
    output['episode_title'] = episode_title
    output['episode_image'] = episode_image
    output['episode_transcript'] = result['text']
    return output


@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_podcast_summary(podcast_transcript):
    import openai
    summary_prompt = """
    You are a copywriter working for a podcast company. Your job is to write a short description of a
    podcast episode that will entice people to listen to it. You have been given a transcript of the
    episode below. Keep the description to 280 characters or less.

  """

    request = summary_prompt + podcast_transcript

    chatOutput = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": request}
        ]
    )

    summary = chatOutput.choices[0].message.content

    return summary


@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_podcast_guest(podcast_transcript):
    import openai
    import wikipedia
    import json

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[{"role": "user", "content": podcast_transcript}],
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
                f"{guest.get('name')} {guest.get('title')} {guest.get('organization')}", auto_suggest=True)
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
        - **Highlight 1:** This is the first highlight I picked
        - **Highlight 2:** This is the second highlight I picked
        - **Highlight 3:** This is the third highlight I picked

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
def process_podcast(url, path):
    output = {}
    podcast_details = get_transcribe_podcast.call(url, path)

    podcast_summary = get_podcast_summary.call(
        podcast_details['episode_transcript'])
    podcast_guests = get_podcast_guest.call(
        podcast_details['episode_transcript'])
    podcast_highlights = get_podcast_highlights.call(
        podcast_details['episode_transcript'])

    output['podcast_details'] = podcast_details
    output['podcast_summary'] = podcast_summary
    output['podcast_guests'] = podcast_guests
    output['podcast_highlights'] = podcast_highlights
    return output


@stub.local_entrypoint()
def test_method(url, path):
    output = {}
    podcast_details = get_transcribe_podcast.call(url, path)
    print("Podcast Summary: ", get_podcast_summary.call(
        podcast_details['episode_transcript']))
    print("Podcast Guest Information: ", get_podcast_guest.call(
        podcast_details['episode_transcript']))
    print("Podcast Highlights: ", get_podcast_highlights.call(
        podcast_details['episode_transcript']))
