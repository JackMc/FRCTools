import os
import json
import csv

from http.client import HTTPConnection

import numpy as np

VERBOSE = True

DEF_HEADERS = {'User-Agent': 'TrueBlue bluealliance Scraper',
               'X-TBA-App-Id': 'frc2994:scouting:v1'}

TBA_BASE = 'www.thebluealliance.com'


def vinput(prompt, validator, default=None):
    if default:
        prompt = prompt + ' [' + default + ']:'
    else:
        prompt += ':'
    print(prompt)
    rawin = input('\t--> ').rstrip()
    val = validator(rawin)
    while rawin and val is False:
        rawin = input('\t--> ').rstrip()
        val = validator(rawin)
    if rawin or not default:
        return rawin
    else:
        return default


def is_integer(s, silence=False):
    try:
        int(s)
        return s
    except ValueError:
        return False


def download_regionals(year):
    conn = HTTPConnection(TBA_BASE)
    conn.request('GET', '/api/v1/events/list?year=' + year,
                 headers=DEF_HEADERS)

    r = conn.getresponse()
    answer = r.read().decode('utf-8')
    return answer


def download_teams(teams_list):
    conn = HTTPConnection(TBA_BASE)
    conn.request('GET', '/api/v1/teams/show?teams=' + ','.join(teams_list),
                 headers=DEF_HEADERS)

    resp = conn.getresponse()
    answer = resp.read().decode('utf-8')
    return answer


def cache_or_get_json(name, func, *args, **kwargs):
    quiet = False

    if 'quiet' in kwargs:
        quiet = kwargs['quiet']

    if not os.path.isdir('cache'):
        os.mkdir('cache')

    filename = 'cache/' + name + '.json'

    if os.path.exists(filename):
        if not quiet and VERBOSE:
            print('Using cached: ' + name)
        return json.loads(open(filename, 'r').read())
    else:
        if not quiet and VERBOSE:
            print('Generating: ' + name)
        value = func(*args)
        open(filename, 'w').write(value)
        return json.loads(value)


def download_regional(key, quiet=False):
    if not quiet:
        print('Downloading event details...')

    conn = HTTPConnection(TBA_BASE)

    conn.request('GET', '/api/v1/event/details?event=' + key,
                 headers=DEF_HEADERS)

    r = conn.getresponse()

    answer = r.read().decode('utf-8')
    return answer


def download_match(m, quiet=False):
    conn = HTTPConnection(TBA_BASE)
    conn.request('GET', '/api/v1/match/details?match=' + m,
                 headers=DEF_HEADERS)

    r = conn.getresponse()
    answer = r.read().decode('utf-8')
    return answer


def make_matches(r):
    match_keys = r.raw_json['matches']

    for m_key in match_keys:
        m_json = cache_or_get_json('match' + m_key, download_match, m_key)

        r.add_match(Match(m_json))


def make_regionals(regionals_gen):
    regionals = []

    for r in regionals_gen:
        r_json = cache_or_get_json('regional' + r['key'], download_regional,
                                   r['key'], True)
        regionals.append(Regional(r_json['key'], r_json['name'], r_json))

    return regionals

teams_cache = {}


def make_teams(teams_text, r=None):
    teams = {}
    if not r:
        teams_json = download_teams(teams_text)
        teams_json = json.loads(teams_json)
    else:
        teams_json = cache_or_get_json(r.key + 'teams', download_teams,
                                       teams_text)

    for t_json in teams_json:
        number = t_json['team_number']

        if number in teams_cache:
            teams[number] = teams_cache[number]
        else:
            t_temp = Team(number, t_json['name'], t_json['website'],
                          t_json['location'])
            teams[number] = t_temp
            teams_cache[number] = t_temp

        for r in t_json['events']:
            teams[number].add_regional(r)

    return teams


class Match:
    QUALS = 0
    ELIMINATION = 1

    def __init__(self, m_json):
        m_json = m_json[0]
        self.red = [int(t.replace('frc', '')) for t in
                    m_json['alliances']['red']['teams']]
        self.blue = [int(t.replace('frc', '')) for t in
                     m_json['alliances']['blue']['teams']]
        self.red_score = m_json['alliances']['red']['score']
        self.blue_score = m_json['alliances']['blue']['score']

        if m_json['competition_level'] == 'Quals':
            self.type = Match.QUALS
        else:
            self.type = Match.ELIMINATION


class Regional:
    def __init__(self, key, name, raw_json):
        self.key = key
        self.name = name
        self.raw_json = raw_json
        self.teams = []
        self.matches = []

    def add_team(self, team):
        self.teams.append(team)

    def add_match(self, m):
        self.matches.append(m)

    def calc_first_stat(self, comparison_fn, method_name):
        teams_sorted = sorted(t.number for t in self.teams)

        # counting_matrix * opr_matrix = totals_matrix

        counting_matrix = np.array([[0 for _ in teams_sorted] for _ in
                                    teams_sorted])

        totals_matrix = np.array([0 for _ in teams_sorted])

        for m in self.matches:
            for r1 in m.red:
                r1_index = teams_sorted.index(r1)
                for r2 in m.red:
                    r2_index = teams_sorted.index(r2)
                    counting_matrix[r1_index][r2_index] += 1
                totals_matrix[r1_index] += comparison_fn(m.red_score,
                                                         m.blue_score)

            for b1 in m.blue:
                b1_index = teams_sorted.index(b1)
                for b2 in m.blue:
                    b2_index = teams_sorted.index(b2)
                    counting_matrix[b1_index][b2_index] += 1
                totals_matrix[b1_index] += comparison_fn(m.blue_score,
                                                         m.red_score)

        try:
            ans_matrix = np.linalg.solve(counting_matrix, totals_matrix)
        except np.linalg.LinAlgError:
            # Discard the error values
            print('Warning: regional ' + self.key +
                  ' needs least-squares approximation (' + method_name + ')')
            ans_matrix = np.linalg.lstsq(counting_matrix, totals_matrix)[0]
        return dict(zip(teams_sorted, ans_matrix))

    def calc_opr(self):
        self.oprs = self.calc_first_stat(lambda us, them: us, 'opr')
        for team_no, opr in self.oprs.items():
            team = teams_cache[team_no]
            team.oprs[self.key] = opr

    def calc_dpr(self):
        self.dprs = self.calc_first_stat(lambda us, them: them, 'dpr')
        for team_no, dpr in self.dprs.items():
            team = teams_cache[team_no]
            team.dprs[self.key] = dpr

    def calc_ccwm(self):
        self.ccwms = self.calc_first_stat(lambda us, them: us - them, 'ccwm')
        for team_no, ccwm in self.ccwms.items():
            team = teams_cache[team_no]
            team.ccwms[self.key] = ccwm


class Team:
    def __init__(self, number, name, website, location):
        self.number = number
        self.oprs = {}
        self.ccwms = {}
        self.dprs = {}
        self.regionals = []

    def add_regional(self, key):
        self.regionals.append(key)

    def attended(self, key):
        return key in self.regionals


# Filter to only necessary regionals to perform calculations (less hammering
# the server and shorter download times)
def filter_regionals(regionals, att_teams):
    newregionals = []
    for r in regionals:
        for t in att_teams.values():
            if t.attended(r.key):
                newregionals.append(r)
                # We don't want to add the same regional many times.
                break
    return newregionals


def interactive_get_regionals():
    year = vinput('Enter the year you wish to check the OPRs of',
                  is_integer)

    regionals_gen = cache_or_get_json('regionals' + year, download_regionals,
                                      year)
    regionals_gen = [r for r in regionals_gen if r['official']]

    return make_regionals(regionals_gen)


def main():
    regionals = interactive_get_regionals()
    teams_important = open(vinput('Enter teams filename',
                                  lambda a: True)).readlines()
    print('Downloading teams from list (can\'t cache)...')
    teams_important = make_teams(['frc' + t.strip() for t in teams_important])

    regionals = filter_regionals(regionals, teams_important)

    teams = {}

    # Complete the regional objects after we've filtered them
    for r in regionals:
        temp_teams = make_teams(r.raw_json['teams'], r)
        r.teams = temp_teams.values()
        teams.update(temp_teams)
        make_matches(r)
        r.calc_opr()
        r.calc_ccwm()
        r.calc_dpr()

    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the headers:
        header_row = ["Team Number"] + (',,,'.join([r.name for r in
                                                    regionals]).split(','))

        writer.writerow(header_row)

        l_teams = teams_important.values()

        rows = [[t.number] for t in l_teams]

        for r in regionals:
            for row in rows:
                row.append(r.oprs[row[0]] if row[0] in r.oprs else '')
                row.append(r.ccwms[row[0]] if row[0] in r.ccwms else '')
                row.append(r.dprs[row[0]] if row[0] in r.dprs else '')

        for row in rows:
            writer.writerow(row)

if __name__ == '__main__':
    main()
