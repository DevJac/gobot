import re
import sys


def name(_args):
    return 'Gobot'


def version(_args):
    return '1'


def protocol_version(_args):
    return '2'


def list_commands(_args):
    return '\n'.join([
        'name',
        'version',
        'protocol_version',
        'list_commands',
        'known_command',
        'quit',
    ])


def known_command(command_name):
    return 'true' if command_name in globals() else 'false'


def quit(_args):
    return ''


logging = False
log_to = '/tmp/gtpdebug'


def log(s):
    if logging:
        with open(log_to, 'a') as f:
            f.write(s + '\n')


def send_response(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    if logging:
        with open(log_to, 'a') as f:
            f.write(f'Sent response: {repr(s)}\n')


def main():
    log('='*85)
    log('='*40 + ' NEW ' + '='*40)
    log('='*85)
    command_re = re.compile(r'(?P<id>[0-9]+)? ?(?P<command_name>\w+) ?(?P<args>.*)')
    for line in sys.stdin:
        log('Received command:' + repr(line))
        try:
            command = re.match(command_re, line)
            response = globals()[command.group('command_name')](command.group('args'))
            response = f"={command.group('id') or ''} {response}\n\n"
            send_response(response)
            if command.group('command_name') == 'quit':
                log('=' * 80 + ' QUIT')
                return
        except Exception:
            log(f'Failed on command: {repr(line)}')
            try:
                log(f'Parsed command as: {command.groups()}')
            except Exception:
                log("Couldn't parse command")
                pass
            raise


if __name__ == '__main__':
    main()
