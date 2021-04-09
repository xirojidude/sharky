
Shader "Skybox/Empty"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _SunDir ("Sun Dir", Vector) = (-.11,.07,0.99,0) 
        _XYZPos ("XYZ Offset", Vector) = (0, 15, -.25 ,0) 
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            uniform sampler2D _MainTex; 
            float4 _SunDir,_XYZPos;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 worldPos : TEXCOORD1;
                float4 vertex : SV_POSITION;   //pos
                float4 screenPos : TEXCOORD2;
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                o.worldPos = mul(unity_ObjectToWorld, v2.vertex);
                o.screenPos  = ComputeScreenPos(o.vertex);
                //o.screenPos.z = -(mul(UNITY_MATRIX_MV, vertex).z * _ProjectionParams.w);     // old calculation, as I used the depth buffer comparision for min max ray march. 

                return o;
            }



         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin



                return fragColor;
            }

            ENDCG
        }
    }
}

#define rotate(a) mat2(cos(a), -sin(a), sin(a), cos(a))
  struct RayHit
{
  bool treeHit;
  bool hit;  
  vec3 hitPos;
  vec3 normal;
  float dist;
  float depth;
  float steps;
  float id;
};

float camRot=1.5;

vec2 cam_Adress = vec2(0.0, 0.0);
vec2 cam_RotAdress = vec2(0.0, 1.0);
float moveSpeed= 0.001; 
vec3 camPos = vec3(0, 0, 1.5);

vec4 read(in sampler2D buffer, in vec2 memPos, vec2 resolution)
{
  return texture(buffer, (memPos+0.5)/resolution, -100.);
}

vec4 GetCloudColor(in vec2 p)
{
  float col = textureLod(iChannel0, p*.19, 0.0).r;
  col -= textureLod(iChannel0, p*.15, 0.0).r;
  col += textureLod(iChannel0, p*.01, 0.0).r;
  col *= textureLod(iChannel0, p*.15, 0.0).r*1.5;
  return vec4(col);
}



float ellipsoid( in vec3 p, in vec3 r )
{
  return (length( p/r ) - 1.0) * min(min(r.x, r.y), r.z);
}


RayHit SkyMarch(in vec3 origin, in vec3 direction)
{
  RayHit result;
  result.treeHit = false;
  float maxDist = 10.10, precis = 0.004;
  float t = 0.0, dist = 0.0, distStep = 0.75;
  vec3 rayPos =vec3(0);

  for ( int i=0; i<64; i++ )
  {
    rayPos =origin+direction*t;

    dist = max(ellipsoid(  rayPos-vec3(0.0, 0., -2.0), vec3( 10.0, 0.5, 10.0)), 
      -ellipsoid( rayPos-vec3(0.0, 0., -2.0), vec3(9.0, 0.49, 9.0)));

    if (abs(dist)<precis || t>maxDist )
    {        
      result.hit=true;
      result.depth = t; 
      result.dist = dist;                              
      result.hitPos = origin+((direction*t)*0.99);   
      result.steps = float(i);
      result.id=1.0;
      break;
    }
    t += dist*distStep;
  }
  if (t>maxDist) {
    result.hit=false;
  }

  return result;
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{  
  if (iFrame>0)
  {
    camPos = read(iChannel3, cam_Adress, iChannelResolution[3].xy).rgb;
    camRot = read(iChannel3, cam_RotAdress, iChannelResolution[3].xy).r;    
   
    if (iMouse.z>0.0) camRot= (iMouse.x - iResolution.x * 0.5)*0.01;    
    camPos.xy-=vec2(cos(camRot-1.5707963268), sin(camRot-1.5707963268))*moveSpeed;
  }


  vec2 uv = fragCoord.xy / iResolution.xy;
  float vig = 0.5 + 0.5*pow( 16.0*uv.x*uv.y*(1.0-uv.x)*(1.0-uv.y), 0.1 );     
  //  vec4 haze =  mix(vec4(0.6,0.7,0.8,1.0),vec4(0.9),  (1.0-uv.y) * 2.0);  
  vec4 color =vec4(0);//haze;



  float camPitch= clamp((iMouse.y - iResolution.y * 0.5)*0.01, 0.0, 1.1);   

  vec2 cloudPos = vec2(float(iTime)*0.01, float(iTime)*0.04)*0.8;
  vec2 scroll = camPos.xy*1.;

  vec3 camOrigin = vec3(0, 0, -2);
  vec2 screenSpace = (2.0*fragCoord.xy-iResolution.xy)/iResolution.y;

  vec3 directionRay = normalize(vec3(screenSpace, 1.0));


  // directionRay.yz *= rotate(camPitch);
  directionRay.xz *= rotate(camRot);




  if (distance(fragCoord, cam_Adress)<=1.0)
  {
    fragColor.rgb = camPos;
  }
  else if (distance(fragCoord, cam_RotAdress)<=1.0)
  {
    fragColor.r = camRot;
  }
    
  if (uv.y>=0.5)
  {
    RayHit marchResult =SkyMarch(camOrigin, directionRay);

    if (marchResult.hit)
    {

      vec4 cloudColor = GetCloudColor(marchResult.hitPos.xz+cloudPos+scroll);
      color = mix(color, mix(color, cloudColor, cloudColor.a), smoothstep(0.0, .12, 1.0/marchResult.dist));
    }

    color.r = mix(color.r, 0.0, 0.80-(distance(screenSpace.y, 0.0)));
  } else
  {
    color.r = texture(iChannel3, vec2(uv.x, 1.0-(uv.y))).a;
  }
  fragColor.a =  clamp(color.r, 0.0, 1.0);
  // fragColor = color;
}



//----------------------------------------------------------------------------------------------------------------------------------------------------------------------


#define rotate(a) mat2(cos(a), -sin(a), sin(a), cos(a))

vec3 light = vec3(2.0, 3.0, 2.0);
float camRot=1.4;
vec2 cam_Adress = vec2(0.0, 0.0);
vec2 cam_RotAdress = vec2(0.0, 1.0);
vec3 camPos = vec3(0, 0, 0);
vec2 scroll;

const vec3 eps = vec3(0.02, 0.0, 0.0);
const int maxRaySteps=360;
const float waterLevel = 9.7;
const float treeDetail = 0.2; //0.0; Change if FPS is too low
const float texWaterLevel = 0.097;
const float treeSpacing = 1.8;
const float treeSpacingFactor = 1.0/1.2;
const float treeSpacingScaler = 0.018*1.2;
const vec2 half_treeSpacing = vec2(0.6);


struct RayHit
{
  bool treeHit;
  bool hit;  
  vec3 hitPos;
  vec3 normal;
  float dist;
  float depth;
  float steps;
  float unclampedHeight;
  float terrainHeight;
  float treeHeight;
};

bool treeHit;
float terrainHeight;
float treeHeight;
float unclampedHeight;
vec2 testPoint;
vec2 test; 
vec2 pointWithoutMod;
float heightMod;

float GetTerrainHeight(in vec2 p)
{
  treeHit=false;
  float col = textureLod(iChannel0, p*0.125, 0.0).r*0.20;
  col -= textureLod(iChannel0, p*0.15, 0.0).r*0.191;
  col += textureLod(iChannel0, p*0.15, 0.0).r*0.1;


  // create trees at the right height level
  if (col<texWaterLevel-0.003 && col>texWaterLevel-0.120)
  {
    testPoint = (p*300.0)+col*12.0;
    test = mod(testPoint, treeSpacing+(col*treeDetail));           
    pointWithoutMod =  testPoint - test; 
    heightMod = textureLod(iChannel0, pointWithoutMod*2.325, 0.0).r;


    treeHeight = (heightMod-pow(treeSpacingFactor*distance(test, half_treeSpacing), 1.0+heightMod))*treeSpacingScaler;      
    treeHeight -= (0.45*distance(col, texWaterLevel-0.0075));

    if (treeHeight>0.002 )
    {
      col -= treeHeight;
      treeHit=true;
    }
  }
  // add stones at beach
  if (col>texWaterLevel-0.01)
  {

    heightMod = textureLod(iChannel0, p*3.0, 0.0).r;


    float stoneHeight = ((heightMod*0.03)-0.015);

    if (stoneHeight>0.001)
    {
      col -= stoneHeight;
    }
  }

  unclampedHeight = col;
  terrainHeight = clamp(col, -1.0, texWaterLevel);
  return terrainHeight;
}



float TerrainDistance( in vec3 p)
{
  return p.y + GetTerrainHeight(vec2(p.xz)+scroll);
}


vec3 CalcNormal( in vec3 pos )
{    
  return normalize( vec3(TerrainDistance(pos+eps.xyy) - TerrainDistance(pos-eps.xyy), 0.5*2.0*eps.x, TerrainDistance(pos+eps.yyx) - TerrainDistance(pos-eps.yyx) ) );
}


float TerrainSoftShadow( in vec3 origin, in vec3 direction )
{
  float res = 2.0;
  float t = 0.0;
  float hardness = 6.0;
  for ( int i=0; i<8; i++ )
  {
    float h = TerrainDistance(origin+direction*t);
    res = min( res, hardness*h/t );
    t += clamp( h, 0.02, 0.10 );
    if ( h<0.002 ) break;
  }
  return clamp( res, 0.0, 1.0 );
}



RayHit TerrainMarch(in vec3 origin, in vec3 direction)
{
  RayHit result;
  result.treeHit = false;
  float maxDist = 1.0, precis = 0.007;
  float t = 0.0, dist = 0.0, distStep = 0.1;
  vec3 rayPos =vec3(0);

  for ( int i=0; i<maxRaySteps; i++ )
  {
    rayPos =origin+direction*t;
    dist = TerrainDistance( rayPos);

    if (abs(dist)<precis || t>maxDist )
    {        
      result.hit = !(t>maxDist);
      result.depth = t; 
      result.dist = dist;                              
      result.hitPos = origin+((direction*t)*0.99);   
      result.steps = float(i);
      result.unclampedHeight=unclampedHeight;
      result.treeHit = treeHit;
      result.terrainHeight = terrainHeight;
      result.treeHeight = treeHeight;
      break;
    }
    t += dist*distStep;
  }

  return result;
}



vec4 read(in sampler2D buffer, in vec2 memPos, vec2 resolution)
{
  return texture(buffer, (memPos+0.5)/resolution, -100.);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{  

  if (iFrame>0)
  {
    camPos = read(iChannel3, cam_Adress, iChannelResolution[3].xy).rgb;
    camRot = read(iChannel3, cam_RotAdress, iChannelResolution[3].xy).r;
  }

  vec2 uv = fragCoord.xy / iResolution.xy;
  if (iMouse.z>0.0) camRot= (iMouse.x - iResolution.x * 0.5)*0.01;   
  //float camPitch= (iMouse.y - iResolution.y * 0.25)*0.01;   

  vec3 camOrigin = vec3(0, 0, -2);
  vec2 screenSpace = (2.0*fragCoord.xy-iResolution.xy)/iResolution.y;
  screenSpace-=0.2;
    
  scroll = camPos.xy;
  vec3 directionRay = normalize(vec3(screenSpace, 1.0));
    
  directionRay.xz *= rotate(camRot);
  directionRay = normalize(directionRay);
    
  vec3 sunPos = normalize(vec3(0.3, 0.7, 1.0));
  vec4 sunColor = vec4(0.80, 0.7, 0.55, 1.0);

  vec4 haze =  mix(vec4(0.9), vec4(0.6, 0.7, 0.8, 1.0), smoothstep(0., 0.6, abs(directionRay.y*2.0))); 
  haze =  mix(haze, vec4(0.6, 0.7, 0.8, 1.0), smoothstep(0., 1.1, abs(directionRay.y*1.1))); 

  vec2 cloudUV = vec2(uv.x, distance(0.0, uv.y) * 1.0); 

  vec4 color = haze;
  vec4 clouds = vec4(texture(iChannel3, cloudUV).a);

  float cloudOpacity = clouds.a;
  vec4 cloudNormal = vec4(vec4(normalize(vec3(texture(iChannel3, cloudUV + vec2(-0.001, 0)).a-texture(iChannel3, cloudUV + vec2(+0.001, 0)).a, texture(iChannel3, cloudUV + vec2(0, -0.001)).a-texture(iChannel3, cloudUV + vec2(0, +0.001)).a, .02)), 1.0).xyz * 0.5 + 0.5, 1.);

  float cloudInt = (cloudNormal.r+cloudNormal.g+cloudNormal.b)*0.333;

  color = max(color, (color+cloudInt)*0.5);


  float diff2= clamp(dot(cloudNormal.xyz, sunPos-directionRay), 0.0, .80);

  clouds*=diff2*sunColor;
  color += color*clouds;


  RayHit marchResult = TerrainMarch(camOrigin, directionRay);


  if (marchResult.hit)
  {
    // get raymarch normal
    marchResult.normal = CalcNormal(marchResult.hitPos);  

    float height =  marchResult.terrainHeight*100.0;
    float realHeight =  marchResult.unclampedHeight*100.0;
    vec4 waterColor = color;
    vec2 waterMotion = vec2(float(iTime)*.005, float(iTime)*.0012);

    // check if area is below water level
    if (height+0.2>waterLevel && !marchResult.treeHit)
    {
      vec4 bottomColor = texture(iChannel2, (marchResult.hitPos.xz+scroll)*22.30);
      float scale1 = 14.0;
      float scale2 = 16.0;
      float scale3 = 22.0;
      vec2 uvCoord = (marchResult.hitPos.xz+scroll+waterMotion)*scale1;

      vec4 wavenormal = vec4(vec4(normalize(vec3(texture(iChannel0, uvCoord
        + vec2(-0.001, 0)).r-texture(iChannel0, uvCoord
        + vec2(+0.001, 0)).r, texture(iChannel0, uvCoord
        + vec2(0, -0.001)).r-texture(iChannel0, uvCoord
        + vec2(0, +0.001)).r, .02)), 1.0).xyz * 0.5 + 0.5, 1.)*0.33333;

      uvCoord = (marchResult.hitPos.xz+scroll+waterMotion)*scale2;

      wavenormal += vec4(vec4(normalize(vec3(texture(iChannel1, uvCoord
        + vec2(-0.001, 0)).r-texture(iChannel0, uvCoord
        + vec2(+0.001, 0)).r, texture(iChannel0, uvCoord
        + vec2(0, -0.001)).r-texture(iChannel0, uvCoord
        + vec2(0, +0.001)).r, .02)), 0.0).xyz * 0.5 + 0.5, 1.)*0.33333;

      uvCoord = (marchResult.hitPos.xz+scroll+waterMotion)*scale3;

      wavenormal += vec4(vec4(normalize(vec3(texture(iChannel2, uvCoord
        + vec2(-0.001, 0)).r-texture(iChannel2, uvCoord
        + vec2(+0.001, 0)).r, texture(iChannel2, uvCoord
        + vec2(0, -0.001)).r-texture(iChannel2, uvCoord
        + vec2(0, +0.001)).r, .02)), 1.0).xyz * 0.5 + 0.5, 1.)*0.33333;


      // set normal to rgb 0,1,0  to do mirror reflections on water surface
      vec3 ref = normalize(reflect(directionRay, vec3(0.0, 1.0, 0.0)));
      RayHit reflectResult = TerrainMarch(marchResult.hitPos + (ref*0.001), ref); 

      float waveDepth = pow((wavenormal.r+wavenormal.g+wavenormal.b)*0.75, 2.0);


      // draw reflected objects and mix with water color
      if (reflectResult.hit==true)
      {
        waterColor *= mix(waterColor, textureLod(iChannel1, (reflectResult.hitPos.xz+scroll)*18.0, 0.0), waveDepth);
      }



      float atten = max(1.0 - dot(wavenormal.r, reflectResult.dist) * 0.001, 0.0)*1.5;


      waterColor*= atten;

      waterColor = mix(waterColor, waterColor*0.12, wavenormal.g);
      waterColor = mix(waterColor, waterColor*1.20, wavenormal.r);
    }

    // terrain color 
    if (marchResult.treeHit==false)
    {

      // texturing
      vec4 texcolor = textureLod(iChannel0, (marchResult.hitPos.xz+scroll)*8.0, 0.0);
      vec4 texcolor2 = mix(mix(texcolor, textureLod(iChannel1, (marchResult.hitPos.xz+scroll)*7.30, 0.0), 0.8), texcolor, pow(texcolor.r, 2.0));
      vec4 texcolor3 = mix(texcolor, textureLod(iChannel2, (marchResult.hitPos.xz+scroll)*22.30, 0.0), 1.4);
      vec4 moss = textureLod(iChannel0, (marchResult.hitPos.xz+scroll)*27.30, 0.0)*1.5*vec4(0.7, 0.8, 0.3, 1.0);


      vec4 finalColor = mix(texcolor, textureLod(iChannel1, (marchResult.hitPos.xz+scroll)*5.90, 0.0), 1.4);
      finalColor= mix(finalColor, textureLod(iChannel1, (marchResult.hitPos.xz+scroll)*4.0, 0.0), 0.5);
      finalColor = mix(finalColor, texcolor2, smoothstep(0.0, 0.3, marchResult.normal.z));

      // apply snow
      finalColor = mix(vec4((texcolor2.r+texcolor2.g+texcolor2.b)+finalColor)*0.85, finalColor, smoothstep(0.04, 0.4, marchResult.normal.x));
      // apply moss
      finalColor = mix(finalColor, moss, smoothstep(0.0, 0.40, pow(texcolor3.r, 3.0))*smoothstep(waterLevel-2.30, waterLevel-0.6, realHeight));


      float diff= clamp(dot(marchResult.normal, normalize(sunPos)), 0.0, 1.0);
      float shadow = TerrainSoftShadow(marchResult.hitPos, normalize(sunPos));
      float amb = clamp( 0.5+0.5*marchResult.normal.r, 0.0, 1.0 )*1.0;

      vec3 lin = vec3(0.0);  
      lin+=diff*shadow;
      lin+=amb*vec3(0.5, 0.5, 0.8)*2.0;


      // apply lightning
      color = finalColor*vec4(lin, 1.0)*sunColor;
      // apply shoreline 
      color = mix(color, texcolor3*1.5*sunColor, smoothstep(waterLevel-0.20, waterLevel-0.2, realHeight));
      // apply water transition
      waterColor = mix((color*waterColor), waterColor, smoothstep(waterLevel, waterLevel+0.4, realHeight));

      // apply water
      color = mix(waterColor+0.24, color, smoothstep(waterLevel, waterLevel-.3, realHeight));

      color=mix(color, color*shadow, 0.34);
    }
    // tree color  
    else
    {
      vec4 snowMask = vec4(textureLod(iChannel1, (marchResult.hitPos.xz+scroll)*12.30, 0.0).r*5.0);
      vec4 pine = textureLod(iChannel0, (marchResult.hitPos.xz+scroll)*127.30, 0.0)*0.6*vec4(0.4, 0.8, 0.1, 1.0);

      color = pine*vec4(1.0/max(1.20, (0.015*marchResult.steps)));

      // apply snow
      color = mix(vec4(snowMask+color)*0.45, color, smoothstep(0.10, 1.0, 1.0/(300.0*marchResult.treeHeight)));
      color = mix(vec4(snowMask+color)*0.45, color, smoothstep(0.04, 0.4, marchResult.normal.x));


      float diff= clamp(dot(marchResult.normal, normalize(sunPos)), 0.0, 1.0);
      float shadow = TerrainSoftShadow(marchResult.hitPos, normalize(sunPos));
      float amb = clamp( 0.5+0.5*marchResult.normal.r, 0.0, 1.0 )*1.0;

      vec3 lin = vec3(0.0);  
      lin+=diff*shadow;
      lin+=amb*vec3(0.5, 0.5, 0.8)*2.0;

      color = color*vec4(lin, 1.0)*sunColor;
    }



    // apply fog
    color =mix(color*(1.5-(0.12*height)), haze, pow(0.95*marchResult.depth, 1.2));
   
      // apply snowy winds
    vec2 winduv= marchResult.hitPos.xz+scroll+vec2(float(iTime)*0.005, float(iTime)*0.02);
    color =mix(color, (color+vec4(textureLod(iChannel0, winduv*0.24, 0.0).r))*0.75, pow(4.0*clamp(marchResult.terrainHeight, 0.0, 1.0), 2.0));
    color =mix(color, (color+vec4(textureLod(iChannel0, winduv*1.24, 0.0).r))*0.75, pow(4.0*clamp(marchResult.terrainHeight, 0.0, 1.0), 2.0));
    color =mix(color, (color+vec4(textureLod(iChannel0, winduv*2.24, 0.0).r))*0.75, pow(4.0*clamp(marchResult.terrainHeight, 0.0, 1.0), 2.0));
  }

  // sun
  float sun = clamp( dot(sunPos, directionRay), 0.0, 1.0 );
  color += vec4(vec3(.9, 0.4, 0.2)*sun*sun*clamp((directionRay.y+0.4)/0.4, 0.0, 0.40), 1.0);

  color*=mix(0.0,1.0, smoothstep(0.0,0.75,float(iTime)));
  fragColor = vec4(pow(color.rgb, vec3(1.0/0.9)), 1.0 ) * (0.5 + 0.5*pow( 16.0*uv.x*uv.y*(1.0-uv.x)*(1.0-uv.y), 0.2 ));
}


